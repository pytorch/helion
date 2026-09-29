"""Distribute a CuTe grid body over the synthetic lane loops it depends on.

``DeviceGridState.wrap_body`` nests every lane loop of a root body around the
whole body.  Each lane loop emulates one SIMD tile axis (a persistent
reduction's synthetic per-thread lanes, a tile block's per-thread elements),
so a statement that reads none of a loop's lane coordinates computes the same
values in every iteration of that loop.  Running it once, in program order,
outside the loop is exactly the tile program's meaning; running it inside
multiplies its work by the trip count.  ``concat2d_dim1_simple`` copies two
full slices of different widths in separate statements: with the second
slice's lane loop around the whole body, the first copy ran once per lane of
the second (27x the traffic at 2048 x (512 + 768)).

``distribute_lane_loops`` places every statement inside exactly the lane loops
whose coordinates it depends on, transitively through the names it reads and
through every definition of a name it writes (a per-lane accumulator keeps
its per-lane initialization).  Each lane loop is materialized once.  A
statement outside a loop is emitted before the loop when a later statement
inside depends on it, and after the loop when it depends on an earlier
statement inside; register and per-tensor memory dependences keep their
source order.  A statement with effects other than plain stores (an atomic,
a barrier, a helper call of unknown purity) is pinned inside every lane loop
and orders every statement that touches memory against itself; a pure
register computation sharing no local name with it (a constant, an index
expression) crosses it, so lane-invariant neighbours may still leave the
loops around it.

A vectorized lane loop also emits statements of its own around the body
(``LaneScope.attached``: the per-thread lane base, the packet loads hoisted
above the constexpr V-loop, the store flushes after it).  Each belongs to
the sites inside the loop that share with it a name the loop defines (the
packet a hoisted load binds, the buffer a flush reads), so a statement moved
across the loop with a register or memory dependence on one of them keeps
its side of those sites.  A statement moved across an outer loop also
crosses the attached statements of every loop nested in that outer loop's
instance.

The transform fails closed.  A body containing compiler markers (lane
reductions, branch-local vector loads: the passes expanding them match the
full nest) keeps the original full nest.  When a loop would need two
instances or a statement has ordering conflicts on both sides of a loop, the
full nest is kept only when it is exact.  That nest repeats every statement
that does not depend on a loop once per iteration of the loop, which is the
tile program's meaning only when the repetition is idempotent and no access
to the same tensor interleaves with it in an order the repetition changes: an
invariant store that a later per-lane store overwrites is applied again by
the next iteration, and an invariant load re-reads what an earlier
iteration's per-lane store wrote.  Such a body is rejected
(``BackendUnsupported``) rather than compiled into a nest that computes
something other than the program.

Memory dependences key on the generated tensor names, as the rest of the
lane-loop lowering does: two kernel arguments that view the same storage are
not ordered against each other.

Names are read through the device function's rename groups.  A value carried
by a nested loop is still written under its loop-output name here (``v_3``)
and only a later pass renames it to the accumulator (``acc``); reading both
as one name keeps the accumulator's initialization in the loops its updates
depend on, once per lane.
"""

from __future__ import annotations

import ast
import collections
import dataclasses
from typing import TYPE_CHECKING

from ... import exc
from ...language.memory_ops import _CUTE_CACHE_LOAD_HELPERS
from ..ast_read_writes import ReadWrites
from ..tile_strategy import _is_proven_relocatable_call
from ..tile_strategy import _memory_write_calls

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping


@dataclasses.dataclass(frozen=True)
class LaneScope:
    """A lane loop (outer to inner order) and the names defined only inside it.

    ``requires`` names the loops this loop's own setup reads (an inner lane's
    index built from an outer lane coordinate): its instance must nest inside
    them.
    """

    lane_var: str
    names: frozenset[str]
    requires: frozenset[str] = frozenset()
    # Statements the loop structure itself emits inside the outer lane loop
    # but outside the constexpr V-loop (see the module docstring).  They are
    # materialized with the loop, never placed; their definitions count as the
    # loop's names and their accesses order the statements moved past them.
    attached: tuple[ast.AST, ...] = ()
    # The loop's own coordinates (its lane variable, its constexpr V-loop
    # variable, its per-thread lane base).  Every attached statement reads
    # them, so unlike the other names the loop defines they do not relate an
    # attached statement to the sites it belongs to.
    coordinates: frozenset[str] = frozenset()


@dataclasses.dataclass
class LanePlacement:
    """One lane loop instance: its statements and nested loops in order."""

    lane_var: str
    items: list[ast.AST | LanePlacement]


_PLAIN_STORE_HELPERS = frozenset({"_cute_store_u16_vec", "_cute_store_u32_vec"})
# Inline-PTX cache-hinted loads and their 8-byte variants.
_PLAIN_LOAD_HELPERS = frozenset(
    name
    for helper in _CUTE_CACHE_LOAD_HELPERS.values()
    for name in (helper, f"{helper}_8b")
)
_RANGE_CALLS = frozenset({"range", "cutlass.range", "cutlass.range_constexpr"})
# The vector type argument of a 16-byte ``cute.arch.load`` and the store
# protocol's register-only conversion of a whole byte packet.
_PURE_CALLS = frozenset({"ir.VectorType.get", "_cute_signed_bitfield_to_bf16_packed"})


def contains_compiler_marker(body: list[ast.AST]) -> bool:
    """Whether a later pass still has to expand a ``_helion_*`` placeholder."""
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id.startswith("_helion_")
        for statement in body
        for node in ast.walk(statement)
    )


def _is_plain_store(call: ast.Call) -> bool:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr in ("store", "__setitem__")
    return isinstance(call.func, ast.Name) and call.func.id in _PLAIN_STORE_HELPERS


def _is_plain_load(call: ast.Call) -> bool:
    return isinstance(call.func, ast.Name) and call.func.id in _PLAIN_LOAD_HELPERS


def _is_list_append(call: ast.Call) -> bool:
    """``buffer.append(value)``: a store protocol collecting a lane's value.

    The list is a name the loop defines, read by the append and by the flush
    after the V-loop, so the append is register work ordered like any other
    statement through the names it reads.
    """
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "append"
        and isinstance(call.func.value, ast.Name)
        and len(call.args) == 1
        and not call.keywords
    )


def _calls_are_movable(node: ast.AST) -> bool:
    return all(
        _is_plain_store(call)
        or _is_plain_load(call)
        or _is_list_append(call)
        or ast.unparse(call.func) in _PURE_CALLS
        or _is_proven_relocatable_call(call, allow_load=True)
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
    )


def _is_movable_iterator(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Call)
        and ast.unparse(node.func) in _RANGE_CALLS
        and all(_calls_are_movable(arg) for arg in [*node.args, *node.keywords])
    )


def _is_movable(node: ast.AST) -> bool:
    """Whether running ``node`` once instead of once per lane preserves it.

    Assignments and expression statements made of proven pure calls (loads
    included), ordinary stores and store buffer appends qualify, as do
    ``range`` loops and branches built only from them.  Anything else pins
    the statement.
    """
    if isinstance(node, ast.For):
        return (
            not node.orelse
            and _is_movable_iterator(node.iter)
            and all(_is_movable(child) for child in node.body)
        )
    if isinstance(node, ast.If):
        return _calls_are_movable(node.test) and all(
            _is_movable(child) for child in [*node.body, *node.orelse]
        )
    if isinstance(node, ast.Pass):
        return True
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr)):
        return _calls_are_movable(node)
    return False


def _tensor_mentions(node: ast.AST) -> collections.Counter[str]:
    """Generated tensor names addressed in ``node``, counted per mention.

    ``t.iterator``, ``t.__setitem__`` and ``t[...]`` address ``t``.  Aliasing
    kernel arguments have distinct names and are not related here.
    """
    mentions: collections.Counter[str] = collections.Counter()
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute):
            if child.attr in ("iterator", "__setitem__") and isinstance(
                child.value, ast.Name
            ):
                mentions[child.value.id] += 1
        elif isinstance(child, ast.Subscript) and isinstance(child.value, ast.Name):
            mentions[child.value.id] += 1
    return mentions


def _tensor_names(node: ast.AST) -> set[str]:
    """Generated tensor names addressed in ``node``."""
    return set(_tensor_mentions(node))


def _store_address(call: ast.Call) -> ast.AST | None:
    """The addressed operand of a plain store; None for a read-modify-write."""
    if isinstance(call.func, ast.Attribute):
        if call.func.attr == "store":
            return call.func.value
        if call.func.attr == "__setitem__":
            return call.func
        return None
    if (
        isinstance(call.func, ast.Name)
        and call.func.id in _PLAIN_STORE_HELPERS
        and call.args
    ):
        return call.args[0]
    return None


def _addressed_tensors(address: ast.AST) -> collections.Counter[str]:
    """The tensor mentions forming ``address`` itself, not those of calls in it."""
    mentions: collections.Counter[str] = collections.Counter()
    pending = [address]
    while pending:
        node = pending.pop()
        if isinstance(node, ast.Call):
            continue
        if isinstance(node, ast.Attribute):
            if node.attr in ("iterator", "__setitem__") and isinstance(
                node.value, ast.Name
            ):
                mentions[node.value.id] += 1
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            mentions[node.value.id] += 1
        pending.extend(ast.iter_child_nodes(node))
    return mentions


def _tensor_accesses(node: ast.AST) -> tuple[set[str], set[str]]:
    """The tensor names ``node`` reads and the ones it writes.

    A plain store writes the tensor its address names; every other mention
    (a load, a store's value, a read-modify-write, an unknown call's
    argument) reads it.
    """
    read = _tensor_mentions(node)
    written: set[str] = set()
    for call in _memory_write_calls(node):
        address = _store_address(call)
        if address is None:
            written |= _tensor_names(call)
        else:
            addressed = _addressed_tensors(address)
            written |= set(addressed)
            read.subtract(addressed)
    return {name for name, count in read.items() if count > 0}, written


def _is_register_only(node: ast.AST) -> bool:
    """Whether ``node`` computes registers from registers, without memory."""
    return isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and all(
        ast.unparse(call.func) in _PURE_CALLS
        or _is_proven_relocatable_call(call, allow_load=False)
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
    )


@dataclasses.dataclass
class _Statement:
    index: int
    node: ast.AST
    reads: frozenset[str]
    writes: frozenset[str]
    tensors: frozenset[str]
    tensors_read: frozenset[str]
    tensors_written: frozenset[str]
    pinned: bool
    register_only: bool
    lanes: set[str] = dataclasses.field(default_factory=set)
    # For a pinned statement: the locally defined names it reads or writes,
    # any of which its unknown effects may mutate in place (a list passed to
    # a helper).
    touched: frozenset[str] = frozenset()
    # For an attached statement (``index`` is -1): the indices of the body
    # statements it belongs to, which give it its place in program order.
    sites: tuple[int, ...] = ()

    def positions(self) -> tuple[int, ...]:
        return (self.index,) if self.index >= 0 else self.sites


def _analyze(index: int, node: ast.AST, renames: Mapping[str, str]) -> _Statement:
    def canonical(names: Iterable[str]) -> frozenset[str]:
        return frozenset(renames.get(name, name) for name in names)

    rw = ReadWrites.from_ast(node)
    read, written = _tensor_accesses(node)
    tensors = canonical(_tensor_names(node))
    return _Statement(
        index,
        node,
        canonical(rw.reads),
        canonical(rw.writes),
        tensors,
        canonical(read),
        canonical(written),
        pinned=not _is_movable(node),
        register_only=not tensors and _is_register_only(node),
    )


def _depends(earlier: _Statement, later: _Statement) -> bool:
    """Whether ``later`` must stay after ``earlier`` (registers, tensor names)."""
    if earlier.pinned or later.pinned:
        if not (earlier.register_only or later.register_only):
            return True
        # A register-only statement crosses an effectful one unless they
        # share a local name the effects may reach.
        pinned, pure = (earlier, later) if earlier.pinned else (later, earlier)
        return bool((pure.reads | pure.writes) & pinned.touched)
    if earlier.writes & (later.reads | later.writes) or earlier.reads & later.writes:
        return True
    return bool(
        earlier.tensors_written & later.tensors
        or later.tensors_written & earlier.tensors
    )


def _conflicts(outside: _Statement, attached: _Statement) -> bool:
    """Whether moving ``outside`` past a loop's ``attached`` statement is observable."""
    if attached.pinned:
        return True
    return bool(
        outside.writes & (attached.reads | attached.writes)
        or outside.reads & attached.writes
        or outside.tensors_written & attached.tensors
        or attached.tensors_written & outside.tensors
    )


def _propagate_lanes(
    statements: list[_Statement],
    scopes: list[LaneScope],
    attached: dict[str, list[_Statement]],
) -> bool:
    """Assign every statement the lane loops it must run inside.

    Returns ``False`` when a loop would have to nest inside a loop that the
    original nest places inside it (its own attached statements read a value
    only an inner loop defines).
    """
    every = {scope.lane_var for scope in scopes}
    order = {scope.lane_var: index for index, scope in enumerate(scopes)}
    requires = {scope.lane_var: set(scope.requires) for scope in scopes}

    def close(lanes: set[str]) -> None:
        pending = list(lanes)
        while pending:
            for lane_var in requires[pending.pop()]:
                if lane_var not in lanes:
                    lanes.add(lane_var)
                    pending.append(lane_var)

    def require(lane_var: str, outer: str) -> bool:
        if outer == lane_var or outer in requires[lane_var]:
            return False
        requires[lane_var].add(outer)
        return True

    for scope in scopes:
        for statement in attached[scope.lane_var]:
            for other in scopes:
                if statement.reads & other.names:
                    require(scope.lane_var, other.lane_var)
    for statement in statements:
        if statement.pinned:
            statement.lanes |= every
        for scope in scopes:
            if statement.reads & scope.names:
                statement.lanes.add(scope.lane_var)
        close(statement.lanes)
    changed = True
    while changed:
        changed = False
        name_lanes: dict[str, set[str]] = {}
        for statement in statements:
            for name in statement.writes:
                name_lanes.setdefault(name, set()).update(statement.lanes)
        # A loop whose hoisted loads read a value defined inside another loop
        # (a relocated row index built from an outer lane coordinate) nests
        # inside that loop.
        for scope in scopes:
            for statement in attached[scope.lane_var]:
                for name in statement.reads:
                    for lane_var in name_lanes.get(name, ()):
                        changed = require(scope.lane_var, lane_var) or changed
        for statement in statements:
            size = len(statement.lanes)
            for name in statement.reads | statement.writes:
                statement.lanes |= name_lanes.get(name, set())
            close(statement.lanes)
            changed = changed or len(statement.lanes) != size
    return all(
        order[outer] < order[lane_var]
        for lane_var, outers in requires.items()
        for outer in outers
    )


def _attribute_sites(
    statements: list[_Statement],
    scopes: list[LaneScope],
    attached: dict[str, list[_Statement]],
) -> None:
    """Record for every attached statement the body statements it belongs to.

    A hoisted load binds a packet the loop defines and its sites read it; a
    flush reads a store buffer that the loop's setup defines or that its site
    binds.  The loop's coordinates are read by every attached statement and
    relate none of them to a site.
    """
    for scope in scopes:
        inside = [s for s in statements if scope.lane_var in s.lanes]
        defined = (scope.names - scope.coordinates).union(*(s.writes for s in inside))
        for statement in attached[scope.lane_var]:
            links = (statement.reads | statement.writes) & defined
            statement.sites = tuple(
                s.index for s in inside if (s.reads | s.writes) & links
            )


def _used_inner(statements: list[_Statement], inner: set[str]) -> set[str]:
    used: set[str] = set()
    for statement in statements:
        used |= statement.lanes & inner
    return used


def _place(
    statements: list[_Statement],
    scopes: list[LaneScope],
    attached: dict[str, list[_Statement]],
) -> list[ast.AST | LanePlacement] | None:
    if not scopes:
        return [statement.node for statement in statements]
    scope, rest = scopes[0], scopes[1:]
    inside = [s for s in statements if scope.lane_var in s.lanes]
    if not inside:
        return _place(statements, rest, attached)
    outside = [s for s in statements if scope.lane_var not in s.lanes]
    if not outside:
        inner = _place(inside, rest, attached)
        return None if inner is None else [LanePlacement(scope.lane_var, inner)]
    inner_vars = {inner_scope.lane_var for inner_scope in rest}
    if _used_inner(inside, inner_vars) & _used_inner(outside, inner_vars):
        # The inner loop would need one instance inside this loop and one
        # outside it.
        return None
    before = {
        s.index
        for s in outside
        if any(t.index > s.index and _depends(s, t) for t in inside)
    }
    after = {
        s.index
        for s in outside
        if any(t.index < s.index and _depends(t, s) for t in inside)
    }
    # The hoisted loads and store flushes of this loop, and of the inner loops
    # nested in its instance, stand for their sites among ``inside``: a
    # conflicting statement keeps its side of all of those sites (of every
    # statement inside when the sites are unknown).
    nested = [
        other
        for lane_var in (scope.lane_var, *_used_inner(inside, inner_vars))
        for other in attached[lane_var]
    ]
    whole = (inside[0].index, inside[-1].index)
    for statement in outside:
        for other in nested:
            if not _conflicts(statement, other):
                continue
            positions = other.sites or whole
            if statement.index < min(positions):
                before.add(statement.index)
            elif statement.index > max(positions):
                after.add(statement.index)
            else:
                return None
    # A statement with no dependence on the loop keeps its side of the loop's
    # last statement; one that depends on a statement already placed after
    # the loop follows it there.
    lead: list[_Statement] = []
    trail: list[_Statement] = []
    for statement in outside:
        forced_after = (
            statement.index in after
            or statement.index > whole[1]
            or any(_depends(t, statement) for t in trail)
        )
        if forced_after:
            if statement.index in before:
                return None
            trail.append(statement)
        else:
            lead.append(statement)
    if _used_inner(lead, inner_vars) & _used_inner(trail, inner_vars):
        return None
    result: list[ast.AST | LanePlacement] = []
    for group in (lead, inside, trail):
        placed = _place(group, rest, attached) if group else []
        if placed is None:
            return None
        if group is inside:
            result.append(LanePlacement(scope.lane_var, placed))
        else:
            result.extend(placed)
    return result


def _relative_order(first: _Statement, second: _Statement) -> str | None:
    """``"before"`` or ``"after"`` when ``first`` is wholly on one side of ``second``."""
    first_positions = first.positions()
    second_positions = second.positions()
    if not first_positions or not second_positions:
        return None
    if max(first_positions) < min(second_positions):
        return "before"
    if min(first_positions) > max(second_positions):
        return "after"
    return None


def _repetition_conflict(
    repeated: _Statement, other: _Statement, other_varies: bool, shared: Iterable[str]
) -> str | None:
    """Why repeating ``repeated`` once per lane iteration around ``other`` is observable.

    ``other`` either varies with the loop (a per-lane access, whose earlier
    and later iterations surround every repetition) or is repeated alongside
    (the same access every iteration).
    """
    for tensor in sorted(shared):
        writes = repeated.pinned or tensor in repeated.tensors_written
        reads = repeated.pinned or tensor in repeated.tensors_read
        other_writes = other.pinned or tensor in other.tensors_written
        other_reads = other.pinned or tensor in other.tensors_read
        if not (writes or other_writes):
            continue
        order = _relative_order(other, repeated)
        if other_varies:
            # Only a store re-applied unchanged survives the surrounding
            # per-lane accesses: after per-lane stores it overwrites in every
            # order, or before per-lane loads that read its value either way.
            exact = (
                writes
                and not reads
                and (
                    (order == "before" and other_writes and not other_reads)
                    or (order == "after" and other_reads and not other_writes)
                )
            )
        else:
            # A read followed by a write of the tensor sees the previous
            # iteration's write.
            exact = (order == "before" and not (other_reads and writes)) or (
                order == "after" and not (reads and other_writes)
            )
        if not exact:
            access = "store to" if writes else "load of"
            other_access = "store to" if other_writes else "load of"
            side = order or "around"
            return (
                f"a lane-invariant {access} {tensor} would repeat {side} "
                f"a{' per-lane' if other_varies else 'nother'} {other_access} it"
            )
    return None


def _inexact_nest(
    statements: list[_Statement],
    scopes: list[LaneScope],
    attached: dict[str, list[_Statement]],
) -> str | None:
    """Why the full lane loop nest computes something other than the program.

    Each loop of the nest repeats every statement that does not depend on it
    once per iteration: the body statements outside it and the attached
    statements of inner loops that read none of its values.
    """
    for depth, scope in enumerate(scopes):
        executed = [(s, scope.lane_var in s.lanes) for s in statements]
        executed.extend((s, True) for s in attached[scope.lane_var])
        varying_names = set(scope.names).union(
            *(s.writes for s, varies in executed if varies)
        )
        for inner in scopes[depth + 1 :]:
            executed.extend(
                (
                    s,
                    scope.lane_var in inner.requires or bool(s.reads & varying_names),
                )
                for s in attached[inner.lane_var]
            )
        for statement, varies in executed:
            if varies:
                continue
            if statement.reads & statement.writes:
                return (
                    f"{ast.unparse(statement.node)} depends on its own result and "
                    f"would repeat once per {scope.lane_var}"
                )
            for other, other_varies in executed:
                if other is statement:
                    continue
                shared = statement.tensors & other.tensors
                if not shared:
                    continue
                reason = _repetition_conflict(statement, other, other_varies, shared)
                if reason is not None:
                    return f"{reason} once per {scope.lane_var}"
    return None


def _require_exact_nest(
    statements: list[_Statement],
    scopes: list[LaneScope],
    attached: dict[str, list[_Statement]],
) -> None:
    reason = _inexact_nest(statements, scopes, attached)
    if reason is not None:
        raise exc.BackendUnsupported(
            "cute", f"the lane loop nest is not the tile program: {reason}"
        )


def _prepare(
    body: list[ast.AST],
    scopes: list[LaneScope],
    rename_groups: Mapping[str, str],
) -> tuple[list[_Statement], list[LaneScope], dict[str, list[_Statement]]]:
    def canonical(names: Iterable[str]) -> frozenset[str]:
        return frozenset(rename_groups.get(name, name) for name in names)

    scopes = [
        dataclasses.replace(
            scope,
            names=canonical(scope.names),
            coordinates=canonical(scope.coordinates),
        )
        for scope in scopes
    ]
    statements = [
        _analyze(index, node, rename_groups) for index, node in enumerate(body)
    ]
    attached = {
        scope.lane_var: [_analyze(-1, node, rename_groups) for node in scope.attached]
        for scope in scopes
    }
    defined = {name for statement in statements for name in statement.writes}
    for scope in scopes:
        defined |= scope.names
    for statement in statements:
        if statement.pinned:
            statement.touched = (statement.reads | statement.writes) & defined
    return statements, scopes, attached


def distribute_lane_loops(
    body: list[ast.AST],
    scopes: list[LaneScope],
    *,
    rename_groups: Mapping[str, str],
) -> list[ast.AST | LanePlacement] | None:
    """Place ``body`` statements inside only the lane loops they depend on.

    ``scopes`` lists the live lane loops from outermost to innermost with the
    names each defines; ``rename_groups`` maps every alias the device
    function will rename to its canonical name.  Returns the emission order
    of statements and ``LanePlacement`` loop instances, or ``None`` when every
    statement depends on every loop (the full nest is already right) or the
    transform cannot prove the redistribution safe and the full nest is
    exact.  Raises ``BackendUnsupported`` when neither holds: the full nest
    would repeat a lane-invariant memory access around per-lane accesses of
    the same tensor (see the module docstring).
    """
    if not body or not scopes or contains_compiler_marker(body):
        return None
    statements, scopes, attached = _prepare(body, scopes, rename_groups)
    nests = _propagate_lanes(statements, scopes, attached)
    _attribute_sites(statements, scopes, attached)
    if not nests:
        _require_exact_nest(statements, scopes, attached)
        return None
    every = {scope.lane_var for scope in scopes}
    if all(statement.lanes == every for statement in statements):
        return None
    placement = _place(statements, list(scopes), attached)
    if placement is None:
        _require_exact_nest(statements, scopes, attached)
    return placement


def check_full_nest(
    body: list[ast.AST],
    scopes: list[LaneScope],
    *,
    rename_groups: Mapping[str, str],
) -> None:
    """Raise ``BackendUnsupported`` unless the full lane loop nest of ``body`` is exact.

    For a caller that falls back to that nest after a placement it cannot
    use; ``distribute_lane_loops`` performs the same check itself whenever it
    keeps the nest.
    """
    if not body or not scopes or contains_compiler_marker(body):
        return
    statements, scopes, attached = _prepare(body, scopes, rename_groups)
    _propagate_lanes(statements, scopes, attached)
    _attribute_sites(statements, scopes, attached)
    _require_exact_nest(statements, scopes, attached)


def definitions_precede_loop(
    placement: list[ast.AST | LanePlacement],
    lane_var: str,
    statements: list[ast.AST],
) -> bool:
    """Whether ``placement`` emits every statement before the loop of ``lane_var``.

    A packet load hoisted into that loop reads the statements' results, so
    they must be bound before the loop and outside it.  True when the loop is
    not materialized at all.
    """
    pending = {id(statement) for statement in statements}

    def walk(items: list[ast.AST | LanePlacement]) -> bool | None:
        for item in items:
            if isinstance(item, LanePlacement):
                if item.lane_var == lane_var:
                    return not pending
                verdict = walk(item.items)
                if verdict is not None:
                    return verdict
            else:
                pending.discard(id(item))
        return None

    verdict = walk(placement)
    return True if verdict is None else verdict
