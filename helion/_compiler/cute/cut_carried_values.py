"""Let a later sweep read the V-loop values carried from its first sweep (CuTe).

Role: device-body pass (pre-rename), right after
``hoist_exact_invariant_divisions`` and before ``jam_grid_lanes``.
``fuse_two_pass_loads`` carries a value a reduction's first sweep computes in
a V-loop (a packet's lanes) into a V-loop of a later sweep, one register
fragment slot per lane, but leaves the later sweep computing it: the read of
the fragment goes in ahead of the statement (``FUSE_CARRIED_READ_ATTR``), the
statement records the read's name (``FUSE_CARRY_ATTR``), and the statements
the carry leaves dead are marked (``FUSE_CARRY_DEAD_ATTR``).  The passes in
between see the sweep as they would without the carry and rewrite those
statements with the rest of it (``fuse_fma`` rounds ``a * b - c`` once), and
the exact division takes the value's kind and selects from them.  Its redo
copy of the sweep recomputes the value from them as they are, which is the
value the carry holds, without keeping the fragment live until the copy's
test.

This pass makes the cut.  Outside the redo copies, each recording statement
takes its read's value, and the marked statements nothing reads any more go
(the names a redo copy reads without binding them count as read).  Inside
the copies the reads go and the marks are dropped: the copy recomputes.

A pass in between may also have rewritten the two sweeps differently:
``fuse_fma`` contracts ``t = a * b; w = t - c`` only where nothing else reads
``t``, so a later sweep's chain can round once where the first sweep's
rounds twice.  So a statement takes its read's value only where its chain
still matches the first sweep's (``_Twins``); otherwise it keeps computing
the value, as the redo copy does.
"""

from __future__ import annotations

import ast
import functools
from typing import TYPE_CHECKING
from typing import cast

from ..ast_read_writes import ReadWrites
from ._ast_pass_utils import _canonical_name
from ._ast_pass_utils import assigned_name
from ._ast_pass_utils import deep_clone
from ._ast_pass_utils import is_name_assign
from ._ast_pass_utils import statement_lists
from .fuse_two_pass_loads import FUSE_CACHE_PREFIX
from .fuse_two_pass_loads import FUSE_CARRIED_READ_ATTR
from .fuse_two_pass_loads import FUSE_CARRY_ATTR
from .fuse_two_pass_loads import FUSE_CARRY_DEAD_ATTR
from .fuse_two_pass_loads import FUSE_CARRY_SOURCE_ATTR
from .fuse_two_pass_loads import FUSE_CARRY_TWIN_ATTR
from .fuse_two_pass_loads import FUSE_TWIN_FIRST_ATTR
from .fuse_two_pass_loads import FUSE_TWIN_LATER_ATTR
from .fuse_two_pass_loads import _load_kind
from .hoist_loop_invariant_recip import EXACT_DIV_REDO_ATTR
from .sweep_values import is_pure_expression

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    from collections.abc import Mapping

_MARKS = (
    FUSE_CARRY_ATTR,
    FUSE_CARRY_DEAD_ATTR,
    FUSE_CARRIED_READ_ATTR,
    FUSE_CARRY_TWIN_ATTR,
    FUSE_CARRY_SOURCE_ATTR,
    FUSE_TWIN_FIRST_ATTR,
    FUSE_TWIN_LATER_ATTR,
)


def _child_blocks(stmt: ast.stmt) -> list[list[ast.stmt]]:
    if isinstance(stmt, (ast.For, ast.While, ast.If)):
        return [stmt.body, stmt.orelse]
    if isinstance(stmt, ast.With):
        return [stmt.body]
    return []


def _is_redo_copy(stmt: ast.stmt) -> bool:
    return isinstance(stmt, ast.If) and getattr(stmt, EXACT_DIV_REDO_ATTR, False)


def _outside_blocks(body: list[ast.stmt]) -> Iterator[list[ast.stmt]]:
    """``body`` and the statement lists nested in it outside the redo
    copies."""
    yield body
    for stmt in body:
        if not _is_redo_copy(stmt):
            for block in _child_blocks(stmt):
                yield from _outside_blocks(block)


def _redo_copies(body: list[ast.stmt]) -> list[ast.If]:
    """The outermost redo copies in ``body``."""
    return [
        cast("ast.If", stmt)
        for block in _outside_blocks(body)
        for stmt in block
        if _is_redo_copy(stmt)
    ]


def _own_reads(stmt: ast.stmt) -> set[str]:
    """What ``stmt`` reads itself: a compound statement's header only."""
    if isinstance(stmt, ast.For):
        part: ast.AST = stmt.iter
    elif isinstance(stmt, (ast.While, ast.If)):
        part = stmt.test
    else:
        part = stmt
    return set(ReadWrites.from_ast(part).reads)


def _free_names(copy: ast.If) -> set[str]:
    """The names ``copy`` reads and does not bind."""
    read = {
        name
        for node in ast.walk(copy)
        if isinstance(node, ast.stmt)
        for name in _own_reads(node)
    }
    bound = {
        node.id
        for node in ast.walk(copy)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    return read - bound


class _Twins:
    """Whether a carried statement still computes the first sweep's value.

    The fuser numbered the group's sweeps and recorded each statement's
    number (``FUSE_TWIN_*_ATTR``).  A carried statement and the first sweep's
    statement of its number must be spelled alike, operand by operand: a
    name reads as the binding that reaches it (the nearest one before it,
    outward), which is a recorded statement's number (and the two statements
    of the number must be spelled alike in turn, but for loads, which the
    fuser proved unchanged), the expression of an unrecorded pure binding or
    load, or the identity of any other binding.  An unrecorded load (one a
    pass in between made) is spelled by its text, which must agree in both
    chains: the passes keep what a chain reads, and the fuser proved the
    loads it numbered unchanged between the sweeps.  The sweeps' own loop
    indices read as their depth.  A name a loop rebinds after its use, or
    that something other than a plain assignment binds, tells nothing."""

    def __init__(self, body: list[ast.stmt], canonical: Callable[[str], str]) -> None:
        self._canonical = canonical
        # Each statement outside the redo copies: its block, its index there
        # and the statement holding the block.
        self._positions: dict[int, tuple[list[ast.stmt], int, ast.stmt | None]] = {}
        self._first: dict[tuple[int, int], list[ast.stmt]] = {}
        self._bound_cache: dict[int, frozenset[str]] = {}
        self._identities: dict[int, int] = {}
        self._index(body, None)

    def _index(self, block: list[ast.stmt], parent: ast.stmt | None) -> None:
        for index, stmt in enumerate(block):
            if _is_redo_copy(stmt):
                continue
            self._positions[id(stmt)] = (block, index, parent)
            for source in getattr(stmt, FUSE_CARRY_SOURCE_ATTR, ()):
                self._first.setdefault(source, []).append(stmt)
            for child in _child_blocks(stmt):
                self._index(child, stmt)

    def _bound(self, stmt: ast.stmt) -> frozenset[str]:
        """The names ``stmt`` binds, anywhere in it (loop targets included)."""
        bound = self._bound_cache.get(id(stmt))
        if bound is None:
            names = set()
            for node in ast.walk(stmt):
                targets = (
                    [node.target]
                    if isinstance(node, ast.For)
                    else [node]
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
                    else []
                )
                names.update(
                    self._canonical(name.id)
                    for target in targets
                    for name in ast.walk(target)
                    if isinstance(name, ast.Name)
                )
            bound = self._bound_cache[id(stmt)] = frozenset(names)
        return bound

    def _binding(self, name: str, stmt: ast.stmt) -> ast.Assign | str | None:
        """The assignment of ``name`` that reaches ``stmt``; ``"unbound"``
        (a kernel argument), ``"loop"`` (the index of a loop around both
        sweeps) or None (nothing certain)."""
        node = stmt
        while True:
            block, index, parent = self._positions[id(node)]
            if isinstance(parent, (ast.For, ast.While)) and any(
                name in self._bound(later) for later in block[index:]
            ):
                return None  # the previous iteration's value
            for prior in reversed(block[:index]):
                if name in self._bound(prior):
                    target = assigned_name(prior)
                    if target is not None and self._canonical(target) == name:
                        return cast("ast.Assign", prior)
                    return None
            if parent is None:
                return "unbound"
            if isinstance(parent, ast.For) and any(
                isinstance(target, ast.Name) and self._canonical(target.id) == name
                for target in ast.walk(parent.target)
            ):
                return "loop"
            node = parent

    def _loops(self, stmt: ast.stmt) -> list[ast.For]:
        """The loops around ``stmt``, innermost first."""
        loops: list[ast.For] = []
        parent = self._positions[id(stmt)][2]
        while parent is not None:
            if isinstance(parent, ast.For):
                loops.append(parent)
            parent = self._positions[id(parent)][2]
        return loops

    def same(self, carried: ast.stmt) -> bool:
        twin = getattr(carried, FUSE_CARRY_TWIN_ATTR, None)
        firsts = self._first.get(twin, []) if twin is not None else []
        if twin is None or len(firsts) != 1 or id(carried) not in self._positions:
            return False
        (first,) = firsts
        later_loops, first_loops = self._loops(carried), self._loops(first)
        # The loops around both sweeps are the same loops.
        while later_loops and first_loops and later_loops[-1] is first_loops[-1]:
            later_loops.pop()
            first_loops.pop()
        if len(later_loops) != len(first_loops) or not all(
            isinstance(loop.target, ast.Name) for loop in (*later_loops, *first_loops)
        ):
            return False
        later_depths, first_depths = (
            {cast("ast.Name", loop.target).id: f"@L{i}" for i, loop in enumerate(loops)}
            for loops in (later_loops, first_loops)
        )
        return self._alike(carried, first, twin[0], later_depths, first_depths, set())

    def _alike(
        self,
        later: ast.stmt,
        first: ast.stmt,
        plan: int,
        later_depths: dict[str, str],
        first_depths: dict[str, str],
        seen: set[tuple[int, int]],
    ) -> bool:
        if (id(later), id(first)) in seen:
            return True
        seen.add((id(later), id(first)))
        if not (is_name_assign(later) and is_name_assign(first)):
            return False
        spelled = self._spell(later, FUSE_TWIN_LATER_ATTR, plan, later_depths)
        twin = self._spell(first, FUSE_TWIN_FIRST_ATTR, plan, first_depths)
        if spelled is None or twin is None or spelled[0] != twin[0]:
            return False
        # Every binding of a number in one chain against every one in the
        # other: a later pass may have rewritten one of several bindings of
        # a number (``fuse_fma`` contracts a product that only one reads).
        return all(
            is_load
            or all(
                self._alike(
                    statement, first_statement, plan, later_depths, first_depths, seen
                )
                for first_statement, _ in twin[1][number]
            )
            for number, bindings in spelled[1].items()
            for statement, is_load in bindings
        )

    def _spell(
        self,
        stmt: ast.Assign,
        attr: str,
        plan: int,
        depths: dict[str, str],
        inlining: frozenset[int] = frozenset(),
    ) -> tuple[str, dict[int, list[tuple[ast.Assign, bool]]]] | None:
        """``stmt``'s value with its names spelled as ``_Twins`` says, and
        the recorded statements it reads, by number (each with whether it
        is a load)."""
        operands: dict[int, list[tuple[ast.Assign, bool]]] = {}
        spelled = deep_clone(stmt.value)
        for node in ast.walk(spelled):
            if not (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)):
                continue
            if node.id in depths:
                node.id = depths[node.id]
                continue
            binding = self._binding(self._canonical(node.id), stmt)
            if binding is None or id(binding) in inlining:
                return None
            if isinstance(binding, str):
                continue  # a kernel argument, or an index of both sweeps
            record = next((r for r in getattr(binding, attr, ()) if r[0] == plan), None)
            if record is not None:
                operands.setdefault(record[1], []).append((binding, record[2]))
                node.id = f"@{record[1]}"
            elif is_pure_expression(binding.value) or _load_kind(binding.value):
                inner = self._spell(
                    binding, attr, plan, depths, inlining | {id(binding)}
                )
                if inner is None:
                    return None
                for number, bindings in inner[1].items():
                    operands.setdefault(number, []).extend(bindings)
                node.id = f"({inner[0]})"
            else:
                node.id = f"${self._identities.setdefault(id(binding), len(self._identities))}"
        return ast.unparse(spelled), operands


def _marked(stmt: ast.stmt) -> bool:
    return bool(
        getattr(stmt, FUSE_CARRY_DEAD_ATTR, False)
        or getattr(stmt, FUSE_CARRIED_READ_ATTR, False)
    )


def _read_carried_values(block: list[ast.stmt], carried: set[int]) -> None:
    """Each statement of ``block`` recording a carried value's read takes the
    read's value if it is in ``carried`` (it still computes the first sweep's
    value), and the read leaves the block.  A statement whose read is no
    longer ahead of it in the block (a pass moved or dropped it) keeps
    computing the value."""
    for stmt in list(block):
        name = getattr(stmt, FUSE_CARRY_ATTR, None)
        if name is None:
            continue
        delattr(stmt, FUSE_CARRY_ATTR)
        if id(stmt) not in carried:
            continue
        position = block.index(stmt)
        read = next(
            (
                earlier
                for earlier in reversed(block[:position])
                if getattr(earlier, FUSE_CARRIED_READ_ATTR, False)
                and assigned_name(earlier) == name
            ),
            None,
        )
        if read is None:
            continue
        assert isinstance(stmt, ast.Assign) and isinstance(read, ast.Assign)
        stmt.value = read.value
        block.remove(read)


def _drop_unread_fragments(body: list[ast.stmt]) -> None:
    """A fragment (``FUSE_CACHE_PREFIX``) no statement reads any more goes,
    with its declaration and its writes: a carried value's whose statements
    all kept computing the value, or any other the passes left unread,
    whose writes are as dead."""
    nodes = list(ast.walk(ast.Module(body=body, type_ignores=[])))
    stored = {
        id(node.value): node.value.id
        for node in nodes
        if isinstance(node, ast.Subscript)
        and isinstance(node.ctx, ast.Store)
        and isinstance(node.value, ast.Name)
    }
    read = {
        node.id
        for node in nodes
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and id(node) not in stored
    }
    unread = {
        name
        for name in set(stored.values()) - read
        if name.startswith(FUSE_CACHE_PREFIX)
    }
    if not unread:
        return

    def writes_unread(stmt: ast.stmt) -> bool:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            return False
        target = stmt.targets[0]
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            return target.value.id in unread
        return isinstance(target, ast.Name) and target.id in unread

    for block in statement_lists(body, outer_first=True):
        kept = [stmt for stmt in block if not writes_unread(stmt)]
        if len(kept) != len(block):
            block[:] = kept or [ast.Pass()]


def _unmark(node: ast.AST) -> None:
    for child in ast.walk(node):
        for mark in _MARKS:
            if hasattr(child, mark):
                delattr(child, mark)


def cut_carried_values(
    body: list[ast.stmt], rename_groups: Mapping[str, str] | None = None
) -> list[ast.stmt]:
    """See the module docstring.  ``rename_groups`` maps a name to the one
    the final rename gives it, so a marked statement whose variable another
    name shares stays."""
    if not any(
        hasattr(node, mark)
        for stmt in body
        for node in ast.walk(stmt)
        for mark in _MARKS
    ):
        return body
    canonical = functools.partial(
        _canonical_name, rename_groups=dict(rename_groups or {})
    )

    # Which values are still the first sweep's: decided before any cut, on the
    # statements as the passes left them.
    twins = _Twins(body, canonical)
    carried = {
        id(stmt)
        for block in _outside_blocks(body)
        for stmt in block
        if getattr(stmt, FUSE_CARRY_ATTR, None) is not None and twins.same(stmt)
    }
    copies = _redo_copies(body)
    for copy in copies:
        for block in statement_lists(copy.body, outer_first=True):
            block[:] = [
                stmt
                for stmt in block
                if not getattr(stmt, FUSE_CARRIED_READ_ATTR, False)
            ]
        _unmark(copy)
    for block in _outside_blocks(body):
        _read_carried_values(block, carried)
    # The marked statements nothing reads go: what the others outside the
    # copies read keeps one alive, and so do the copies' free names and what
    # a kept one reads.
    marked = [
        (block, stmt)
        for block in _outside_blocks(body)
        for stmt in block
        if _marked(stmt)
    ]
    live = {
        canonical(name)
        for block in _outside_blocks(body)
        for stmt in block
        if not _marked(stmt) and not _is_redo_copy(stmt)
        for name in _own_reads(stmt)
    }
    for copy in copies:
        live |= {canonical(name) for name in _free_names(copy)}
    # A marked statement that binds no plain name (a later pass made it a
    # tuple or augmented assignment) stays, and so does what it reads.
    kept = {id(stmt) for _block, stmt in marked if assigned_name(stmt) is None}
    live |= {
        canonical(name)
        for _block, stmt in marked
        if id(stmt) in kept
        for name in _own_reads(stmt)
    }
    changed = True
    while changed:
        changed = False
        for _block, stmt in marked:
            target = assigned_name(stmt)
            if id(stmt) in kept or target is None or canonical(target) not in live:
                continue
            kept.add(id(stmt))
            live |= {canonical(name) for name in _own_reads(stmt)}
            changed = True
    for block, stmt in marked:
        if id(stmt) not in kept:
            block.remove(stmt)
            if not block:
                block.append(ast.Pass())
    _drop_unread_fragments(body)
    for stmt in body:
        _unmark(stmt)
    return body
