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
source order.  An atomic read-modify-write whose address names its tensor is
placed like a store: it depends on the lanes its address and value read and
its tensor orders it against every other access of that tensor.  It is never
repeated: when it runs inside a loop its values ignore (below), it is pinned
to the loop's first lane if the atomic is uniform along that loop's tile axis
(``cute/atomic_ops.py`` records those loops on the call: the leader thread's
first element is the tile's first, always in range), and the body is rejected
otherwise.  A relaxed
atomic crosses accesses of other tensors; an acquire or release one orders
every memory access against itself.  A statement with effects the pass cannot
see whole (a barrier, a helper call of unknown purity, an atomic through a
pointer alias) is pinned inside every lane loop and orders every statement
that touches memory against itself; a pure register computation sharing no
local name with it (a constant, an index expression) crosses it, so
lane-invariant neighbours may still leave the loops around it.  A statement
binding a CTA or cluster reduce (an RMSNorm row's shared-memory or warp sum
beside the lane loop of a half-row slice) whose operands the pass sees whole
(register computations and plain loads) is placed by those operands instead
(``statements._assigns_a_collective_reduce``): every thread still reaches
it equally often, once outside a loop its operands ignore.

A vectorized lane loop also emits statements of its own around the body
(``LaneScope.attached``: the per-thread lane base, the packet loads hoisted
above the constexpr V-loop, the store flushes after it).  Each belongs to
the sites inside the loop that share with it a name the loop defines (the
packet a hoisted load binds, the buffer a flush reads), so a statement moved
across the loop with a register or memory dependence on one of them keeps
its side of those sites.  A statement moved across an outer loop also
crosses the attached statements of every loop nested in that outer loop's
instance.  A loop whose attached statements read an outer loop's values (a
packet load addressed by the outer lane's row index) nests inside that loop,
and so does every statement inside it, whether or not its own values change
with the outer loop; so does a loop that shares a statement with an outer
loop, since each loop is materialized once (the statements that need only
the inner loop then run inside the outer one too, rather than the whole
body running inside every loop).  A partial tile guards every access with
the masks of all its axes, so a statement whose address never changes with a
loop still reads the loop's mask and is placed inside it (a ``tile.begin``
access carries no lane mask).  Each statement therefore carries two lane
sets: the loops it is placed in, and the loops its values depend on, which
ignore the masks (a mask only skips the statement in the iterations past the
tile's end).

The transform fails closed.  A body containing compiler markers (lane
reductions, branch-local vector loads: the passes expanding them match the
full nest) keeps the original full nest.  When a statement has ordering
conflicts on both sides of a loop, or a loop's single instance would have to
sit on both sides of another, the
full nest is kept.  Whichever nest is emitted, the placement or the original
one, is checked the same way: each of its loops repeats every statement it
runs whose values do not depend on it once per iteration, which is the tile
program's meaning only when the repetition is idempotent and no access to
the same tensor interleaves with it in an order the repetition changes: a
lane-invariant atomic accumulates once per iteration, an invariant store that
a later per-lane store overwrites is applied again by the next iteration, and
an invariant load re-reads what an earlier iteration's per-lane store wrote.
A device loop (or a ``while`` loop) of the body whose values depend on the
lane loop runs whole once per lane, so the accesses inside it whose values
do not are repeated with every iteration of it between two repetitions: a
load of a tensor the loop stores to, or a store to one it loads or stores
per lane, is observable there in any order, the first pass's later
iterations coming before the second pass's earlier ones; against the
statements around the loop each repetition sits at the loop's place, as a
lane-invariant statement of the list would.  A branch whose values depend
on the lane loop is emitted whole inside it too (its sides are not
distributed): a statement of its sides whose values do not is repeated once
per lane between the per-lane statements of the passes, and is checked like
a statement of the loop's own list (a first-column load beside the per-lane
update of its row re-reads what lane 0 stored).
Such a body is rejected (``BackendUnsupported``) rather than compiled into a
nest that computes something other than the program, and an atomic placed in
a loop its values ignore is pinned or rejected on every path.  A placement is
not exact by construction: the statements of an inner loop nested inside an
outer one by its hoisted packets run inside the outer loop whether or not
their values change with it, so a body that would be rejected as the full
nest is rejected as a placement too.
"""

from __future__ import annotations

import ast
import dataclasses
import logging
import operator
from typing import TYPE_CHECKING
from typing import cast

from .... import exc
from ...ast_extension import create
from ...ast_extension import expr_from_string
from ...ast_read_writes import HELION_ATOMIC_UNIFORM_LANES_ATTR
from ...ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from ...ast_read_writes import ReadWrites
from .markers import _HELION_LANE_REDUCE_MARKER
from .statements import _METADATA_ATTRS
from .statements import LOGGER_NAME
from .statements import LanePlacement
from .statements import LaneScope
from .statements import _analyze
from .statements import _carried_writes
from .statements import _conflicts
from .statements import _conjuncts
from .statements import _depends
from .statements import _is_atomic_call
from .statements import _pointer_strategy_tensor
from .statements import _propagate_dataflow_lanes
from .statements import _propagate_dependence
from .statements import _propagate_lanes
from .statements import _simple_statements
from .statements import _Statement
from .statements import _structural_markers
from .statements import _tensor_names
from .statements import _uniform_lanes_of

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import Mapping
    from collections.abc import Sequence


log = logging.getLogger(LOGGER_NAME)


def _repeated_carry(statement: _Statement, names: frozenset[str], lane_var: str) -> str:
    return (
        f"{ast.unparse(statement.node)} updates the loop-carried "
        f"{', '.join(sorted(names))} and would repeat once per {lane_var}"
    )


def _repeated_atomic(statement: _Statement, lane_var: str) -> str:
    tensors = ", ".join(sorted(statement.tensors_written))
    return f"a lane-invariant atomic on {tensors} would repeat once per {lane_var}"


def _pinned(node: ast.AST, predicate: str) -> ast.AST | None:
    """``node`` issuing its atomic only under ``predicate``.

    The atomic's expression statement, alone or as the one statement of a
    chain of branches (the codegen's mask guard inside a user branch), is
    guarded by ``predicate``; the original statement is left as it was.
    None for any other statement: one binding the atomic's result (the other
    lanes would read the placeholder) or one holding other work, which the
    predicate would skip along with it.
    """
    test = expr_from_string(predicate)
    assert isinstance(test, ast.expr)
    if isinstance(node, ast.Expr):
        return create(ast.If, test=test, body=[node], orelse=[])
    if (
        isinstance(node, ast.If)
        and len(node.body) == 1
        and all(isinstance(child, ast.Pass) for child in node.orelse)
    ):
        if isinstance(node.body[0], ast.Expr):
            guarded = create(ast.BoolOp, op=ast.And(), values=[node.test, test])
            return create(ast.If, test=guarded, body=node.body, orelse=node.orelse)
        inner = _pinned(node.body[0], predicate)
        if inner is not None:
            assert isinstance(inner, ast.stmt)
            return create(ast.If, test=node.test, body=[inner], orelse=node.orelse)
    return None


def _unpinnable(node: ast.AST, lane_vars: list[str]) -> str:
    """Why ``_pinned`` declined ``node`` for the atomics uniform along ``lane_vars``."""
    atomics = [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and _is_atomic_call(call)
        and _uniform_lanes_of(call) & set(lane_vars)
    ]
    if any(
        isinstance(child, ast.Assign) and any(child.value is call for call in atomics)
        for child in ast.walk(node)
    ):
        return (
            "the result of an atomic issued by one lane is not shared with the "
            "other lanes of its tile axis"
        )
    tensors = ", ".join(sorted(set().union(*(_tensor_names(call) for call in atomics))))
    return (
        f"a lane-invariant atomic on {tensors} issued at the first "
        f"{', '.join(lane_vars)} would skip the rest of its statement in the "
        "other lanes"
    )


def _varying_pin(statement: _Statement, lane_vars: list[str]) -> str:
    """Why ``statement`` cannot be issued at the first lane of ``lane_vars``."""
    tensors = ", ".join(sorted(statement.tensors_written))
    return (
        f"a lane-invariant atomic on {tensors} sits in a statement whose values "
        f"vary with {', '.join(lane_vars)}, so it cannot be issued at the first "
        "lane only"
    )


def _pin_conflict(
    statement: _Statement,
    lane_var: str,
    statements: list[_Statement],
    scopes: list[LaneScope],
    attached: dict[str, list[_Statement]],
    *,
    full_nest: bool,
) -> str | None:
    """Why issuing ``statement`` at the first iteration of ``lane_var``'s loop is observable.

    Pinned, the atomic runs after the first iteration's earlier accesses and
    before every later iteration's.  That is the tile program only when every
    other access of its tensors inside the loop varies with the loop and
    follows the atomic in program order, so that all of them see its effect as
    the program's do.  A per-lane access the atomic follows in program order
    is still applied by the later iterations after it; an access repeated by
    the loop sits in the same position every iteration; a load hoisted above
    the loop's V-loop precedes the atomic in the pinned iteration even when
    its site follows it; unknown effects may be any of these.  A fence (an
    atomic that is not relaxed) orders every memory access against itself, so
    pinned it leaves the other iterations' accesses unordered.
    """
    depth = next(
        index for index, scope in enumerate(scopes) if scope.lane_var == lane_var
    )
    inner = {scope.lane_var for scope in scopes[depth + 1 :]}
    inside = [
        other
        for other in statements
        if other is not statement and (full_nest or lane_var in other.lanes)
    ]
    used = inner if full_nest else _used_inner([*inside, statement], inner)
    hoisted = [other for loop in (lane_var, *sorted(used)) for other in attached[loop]]
    tensors = ", ".join(sorted(statement.tensors_written))

    def access(other: _Statement) -> str:
        return "store to" if other.tensors_written & statement.tensors else "load of"

    for other in [*inside, *hoisted]:
        if other.pinned:
            return (
                f"a lane-invariant atomic on {tensors} issued at the first "
                f"{lane_var} would run beside a statement with unknown effects"
            )
        if statement.fence and other.tensors:
            return (
                f"a lane-invariant atomic on {tensors} issued at the first "
                f"{lane_var} would not order the other lanes' accesses of "
                + ", ".join(sorted(other.tensors))
            )
        if not statement.tensors & other.tensors:
            continue
        if other.index < 0:
            return (
                f"a lane-invariant atomic on {tensors} issued at the first "
                f"{lane_var} would run beside the loop's hoisted {access(other)} it"
            )
        if lane_var not in other.dataflow_lanes:
            return (
                f"a lane-invariant atomic on {tensors} issued at the first "
                f"{lane_var} would run beside a repeated {access(other)} it"
            )
        if other.index < statement.index:
            return (
                f"a lane-invariant atomic on {tensors} issued at the first "
                f"{lane_var} would follow a per-lane {access(other)} it in that "
                "lane only"
            )
    return None


def _pin_repeated_atomics(
    body: list[ast.AST],
    statements: list[_Statement],
    scopes: list[LaneScope],
    attached: dict[str, list[_Statement]],
    renames: Mapping[str, str],
    *,
    full_nest: bool,
) -> None:
    """Pin each atomic to the first lane of the uniform loops it runs in; reject the rest.

    An atomic runs inside every loop of ``lanes`` (inside every loop when the
    full nest is kept) and each loop is materialized once, so a loop its
    values ignore repeats it per iteration.  Along a loop whose tile axis the
    atomic is uniform on, issuing it at the loop's first lane and first
    vector lane is the tile program: the leader thread's first element is the
    tile's first along that axis, always in range, and a value that merely
    reads the loop's mask is the tile's value there.  The guard makes the
    statement read the loop's coordinates, so it counts as varying with the
    loop from here on.  Any other repeated loop rejects the body, as does a
    pin that reorders the atomic against other accesses of its tensors
    (``_pin_conflict``), a statement ``_pinned`` cannot guard, and a
    statement whose values vary with the pinned loop (a compound statement
    holding per-lane work beside the uniform atomic, a branch on a per-lane
    condition): its atomic is uniform along the loop but the statement is
    not, and one lane's iteration would stand for all of them.  A compound
    statement is decided per atomic call: any call uniform along a loop it
    runs in without varying pins the whole statement.  ``body`` is updated in
    place.
    """
    order = {scope.lane_var: index for index, scope in enumerate(scopes)}
    by_lane = {scope.lane_var: scope for scope in scopes}
    for statement in statements:
        if not statement.atomic:
            continue
        runs_in = set(order) if full_nest else statement.lanes
        repeated = runs_in - statement.dataflow_lanes - statement.uniform_lanes
        if repeated:
            raise inexact_nest_error(
                _repeated_atomic(statement, min(repeated, key=order.__getitem__))
            )
        pinned = (runs_in & statement.partly_uniform_lanes) - statement.pinned_lanes
        if not pinned:
            continue
        lane_vars = sorted(pinned, key=order.__getitem__)
        for lane_var in lane_vars:
            reason = _pin_conflict(
                statement, lane_var, statements, scopes, attached, full_nest=full_nest
            )
            if reason is not None:
                raise inexact_nest_error(reason)
        predicate = " and ".join(by_lane[lane_var].first_lane for lane_var in lane_vars)
        node = _pinned(statement.node, predicate)
        if node is None:
            raise exc.BackendUnsupported("cute", _unpinnable(statement.node, lane_vars))
        varying = pinned & statement.dataflow_lanes
        if varying:
            raise inexact_nest_error(
                _varying_pin(statement, sorted(varying, key=order.__getitem__))
            )
        body[statement.index] = node
        statement.node = node
        statement.reads |= frozenset(
            renames.get(name, name)
            for name in ReadWrites.from_ast(ast.parse(predicate, mode="eval")).reads
        )
        statement.lanes |= pinned
        statement.dataflow_lanes |= pinned
        statement.pinned_lanes |= pinned
        # A later analysis of the same body (``check_full_nest`` after an
        # abandoned placement) sees the guard as the atomic's own dependence
        # on these loops and must not pin them again.
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and _is_atomic_call(call):
                setattr(
                    call,
                    HELION_ATOMIC_UNIFORM_LANES_ATTR,
                    _uniform_lanes_of(call) - pinned,
                )


def pin_atomics_left_in_lane_loops(body: list[ast.AST]) -> None:
    """Issue every atomic uniform along an enclosing lane loop at its first lane.

    Runs on the nest ``tile_strategy.split_lane_loop_reductions`` rebuilds for
    a body holding ``_helion_lane_reduce`` markers.  The split hoists the
    statements that read no lane out of its per-lane passes, but an atomic
    uniform along a loop whose statement reads a lane's copy of a value (a
    cached load) stays inside it and would run once per iteration.  Pinned
    before the split, an atomic would read the lane through its guard and
    lose the hoist.  Each such statement directly inside a lane loop's body
    is pinned (``_pin_uniform_atomics``); one nested deeper (in a branch or a
    device loop) rejects the body.  ``body`` is updated in place.
    """

    def visit(statements: list[ast.AST], loops: list[ast.For]) -> None:
        lane_vars = [ast.unparse(loop.target) for loop in loops]
        direct = bool(loops) and statements is loops[-1].body
        for position, node in enumerate(statements):
            if isinstance(node, ast.For) and getattr(
                node, HELION_LANE_LOOP_VAR_ATTR, None
            ) == ast.unparse(node.target):
                visit(cast("list[ast.AST]", node.body), [*loops, node])
                continue
            guarded = _pin_uniform_atomics(
                node,
                {
                    lane_var: f"{lane_var} == 0" if direct else ""
                    for lane_var in lane_vars
                },
                {ast.unparse(loop.target): loop.body for loop in loops},
            )
            if guarded is not None:
                statements[position] = guarded
                continue
            for field in ("body", "orelse", "finalbody"):
                children = getattr(node, field, None)
                if isinstance(children, list) and children:
                    visit(children, loops)

    visit(body, [])


def _pin_atomics_beside_compiler_markers(
    body: list[ast.AST], scopes: list[LaneScope], markers: set[str]
) -> None:
    """Pin the atomics of a body the distribution leaves to a later pass.

    A body holding ``_helion_*`` placeholders (``markers``: a collective
    matmul's ``_helion_pending_collective_mma``) is neither redistributed nor
    checked; its caller wraps the full nest of ``scopes`` around it.  Each
    top-level statement of ``body`` with an atomic uniform along one of those
    loops (a tile counter beside the matmul's epilogue) is pinned to their
    first lanes (``_pin_uniform_atomics``, ``LaneScope.first_lane``).  A body
    holding only ``_helion_lane_reduce`` markers is left to the split that
    rebuilds its loops (``pin_atomics_left_in_lane_loops``).  ``body`` is
    updated in place.
    """
    if markers <= {_HELION_LANE_REDUCE_MARKER}:
        return
    first_lanes = {scope.lane_var: scope.first_lane for scope in scopes}
    bodies = dict.fromkeys(first_lanes, body)
    for index, node in enumerate(body):
        guarded = _pin_uniform_atomics(node, first_lanes, bodies)
        if guarded is not None:
            body[index] = guarded


def _pin_uniform_atomics(
    node: ast.AST,
    first_lanes: Mapping[str, str],
    bodies: Mapping[str, Sequence[ast.AST]],
) -> ast.AST | None:
    """``node`` with its uniform atomics issued at their loops' first lanes.

    ``first_lanes`` maps the variables of the lane loops around ``node``,
    outermost first, to their first-lane tests ("" where ``node`` cannot take
    one), and ``bodies`` maps them to the statements a pin to that loop may
    reorder ``node`` against.  None when no atomic of ``node`` is recorded as
    uniform along those loops (``HELION_ATOMIC_UNIFORM_LANES_ATTR``).  A
    missing test, a statement ``_pinned`` cannot guard, or a pin
    ``_rebuilt_pin_conflict`` finds observable rejects the body.  The pinned
    atomics are uniform along those loops no longer.
    """
    calls = [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and _is_atomic_call(call)
        and _uniform_lanes_of(call) & first_lanes.keys()
    ]
    if not calls:
        return None
    pinned = [
        lane_var
        for lane_var in first_lanes
        if any(lane_var in _uniform_lanes_of(call) for call in calls)
    ]
    tests = [first_lanes[lane_var] for lane_var in pinned]
    guarded = _pinned(node, " and ".join(tests)) if all(tests) else None
    if guarded is None:
        raise exc.BackendUnsupported("cute", _unpinnable(node, pinned))
    reason = _rebuilt_pin_conflict(node, bodies[pinned[0]], pinned[0])
    if reason is not None:
        raise exc.BackendUnsupported("cute", reason)
    for call in calls:
        setattr(
            call,
            HELION_ATOMIC_UNIFORM_LANES_ATTR,
            _uniform_lanes_of(call) - set(pinned),
        )
    return guarded


def _rebuilt_pin_conflict(
    node: ast.AST, loop_body: Sequence[ast.AST], lane_var: str
) -> str | None:
    """Why issuing the atomic statement ``node`` at its loop's first lane is observable.

    Conservative ``_pin_conflict`` for the pins of a body a later pass still
    expands (``pin_atomics_left_in_lane_loops``,
    ``_pin_atomics_beside_compiler_markers``): any other access in
    ``loop_body`` of a tensor the atomic writes, or of any tensor for a
    fence, may be another lane's that the program orders after (or before)
    the atomic.  A call of unknown effects reaches only the tensors it
    mentions, except that a fence orders its unknown effects too.
    """
    atomic = _analyze(-1, node, {})
    tensors = ", ".join(sorted(atomic.tensors_written))

    def others(statements: Sequence[ast.AST]) -> Iterator[ast.AST]:
        for statement in statements:
            if statement is node:
                continue
            if any(child is node for child in ast.walk(statement)):
                for field in ("body", "orelse", "finalbody"):
                    yield from others(getattr(statement, field, None) or [])
            else:
                yield statement

    for statement in others(loop_body):
        other = _analyze(-1, statement, {})
        if atomic.fence and (other.tensors or other.pinned):
            return (
                f"a lane-invariant atomic on {tensors} issued at the first "
                f"{lane_var} would not order the other lanes' accesses around it"
            )
        if other.tensors & atomic.tensors_written:
            return (
                f"a lane-invariant atomic on {tensors} issued at the first "
                f"{lane_var} would run beside another access of its tensors "
                "in the loop"
            )
    return None


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


def _repeatable(statement: _Statement, written: frozenset[str]) -> bool:
    """Whether running ``statement`` once per lane of a loop its values ignore
    is the tile program by construction: a register computation or a load
    of tensors no statement of the body writes or reaches through an unknown
    call (``written``), with no store, atomic, fence or unknown effect of
    its own."""
    return not (
        statement.pinned
        or statement.atomic
        or statement.fence
        or statement.tensors_written
        or statement.tensors & written
    )


def _repeat_inside(
    statement: _Statement,
    trail: list[_Statement],
    statements: list[_Statement],
    lane_var: str,
    attached: dict[str, list[_Statement]],
    universe: list[_Statement],
) -> bool:
    """Move ``statement`` and the ``trail`` statements it depends on into the
    lane loop ``lane_var`` when every one of them is ``_repeatable`` and no
    statement left outside the loop reads a name they define.

    Returns ``False``, moving nothing, otherwise.  ``_place`` restarts the
    scope's placement with the enlarged inside (each move strictly grows it,
    so the restarts end).  ``written`` counts the tensors the loops' attached
    statements (hoisted loads, store flushes) write as well as the body's.
    A name a moved statement defines is visible only inside the loop: a
    reader anywhere in the body (``universe``, every group of every scope)
    that stays outside would be emitted after the loop reading an undefined
    name, so such a move is refused and the placement abandoned."""
    written = frozenset().union(
        *(other.tensors_written | other.opaque for other in statements),
        *(
            other.tensors_written | other.opaque
            for others in attached.values()
            for other in others
        ),
    )
    moved: list[_Statement] = []
    pending = [statement]
    while pending:
        current = pending.pop()
        if current in moved:
            continue
        if not _repeatable(current, written):
            return False
        moved.append(current)
        pending.extend(
            other for other in trail if other not in moved and _depends(other, current)
        )
    defined = frozenset().union(*(other.writes for other in moved))
    for other in universe:
        if other in moved or lane_var in other.lanes:
            continue
        if other.reads & defined:
            log.debug(
                "lane placement abandoned at %s: statement %d reads %s outside "
                "the loop the definition would move into",
                lane_var,
                other.index,
                sorted(other.reads & defined),
            )
            return False
    for other in moved:
        other.lanes.add(lane_var)
    return True


def _place(
    statements: list[_Statement],
    scopes: list[LaneScope],
    attached: dict[str, list[_Statement]],
    universe: list[_Statement] | None = None,
) -> list[ast.AST | LanePlacement] | None:
    # ``universe`` is the whole body: the groups a scope splits off recurse
    # with their own ``statements`` and still need to see every reader.
    if universe is None:
        universe = statements
    if not scopes:
        return [statement.node for statement in statements]
    scope, rest = scopes[0], scopes[1:]
    inside = [s for s in statements if scope.lane_var in s.lanes]
    if not inside:
        return _place(statements, rest, attached, universe)
    outside = [s for s in statements if scope.lane_var not in s.lanes]
    if not outside:
        inner = _place(inside, rest, attached, universe)
        return None if inner is None else [LanePlacement(scope.lane_var, inner)]
    inner_vars = {inner_scope.lane_var for inner_scope in rest}
    # ``_propagate_lanes`` nests every loop a statement shares with this one
    # inside it, so no inner loop is used both inside and outside.
    assert not _used_inner(inside, inner_vars) & _used_inner(outside, inner_vars)
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
                log.debug(
                    "lane placement abandoned at %s: statement %d conflicts with "
                    "an attached statement of the loop inside its span %s",
                    scope.lane_var,
                    statement.index,
                    positions,
                )
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
                # Needed before the loop by a statement inside and after it
                # by the loop's ordering: a mask computed from a load that a
                # pinned statement inside (a call the pass cannot see
                # through, such as an is_pure=False inline asm) orders after
                # itself, feeding a store inside.  When the
                # statement and the trail it depends on only compute registers
                # and load tensors nothing in the body writes, running them
                # inside the loop once per lane is the tile program by
                # construction; the emitted nest is checked like any other.
                if not _repeat_inside(
                    statement, trail, statements, scope.lane_var, attached, universe
                ):
                    log.debug(
                        "lane placement abandoned at %s: statement %d must sit on "
                        "both sides of the loop (%s)",
                        scope.lane_var,
                        statement.index,
                        ast.unparse(statement.node)[:160],
                    )
                    return None
                return _place(statements, scopes, attached, universe)
            trail.append(statement)
        else:
            lead.append(statement)
    if _used_inner(lead, inner_vars) & _used_inner(trail, inner_vars):
        log.debug(
            "lane placement abandoned at %s: an inner loop is used both before "
            "and after it",
            scope.lane_var,
        )
        return None
    result: list[ast.AST | LanePlacement] = []
    for group in (lead, inside, trail):
        placed = _place(group, rest, attached, universe) if group else []
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
            other_access = (
                "atomic on"
                if other.atomic
                else "store to"
                if other_writes
                else "load of"
            )
            side = order or "around"
            return (
                f"a lane-invariant {access} {tensor} would repeat {side} "
                f"a{' per-lane' if other_varies else 'nother'} {other_access} it"
            )
    return None


def _pointer_tensors(node: ast.AST, renames: Mapping[str, str]) -> frozenset[str]:
    """The tensors ``node`` addresses through their pointers (``t.iterator``,
    or the pointer indexing strategy's ``tl.load(t + offset, ...)``).

    Those are the kernel's tensors as the memory-op codegen loads and stores
    them; a shared-memory view or register array an emitter builds itself is
    indexed by subscript.  A metadata query (``t.iterator.alignment``) is no
    access.
    """
    queried = {
        id(child.value)
        for child in ast.walk(node)
        if isinstance(child, ast.Attribute) and child.attr in _METADATA_ATTRS
    }
    tensors = {
        renames.get(child.value.id, child.value.id)
        for child in ast.walk(node)
        if isinstance(child, ast.Attribute)
        and child.attr == "iterator"
        and isinstance(child.value, ast.Name)
        and id(child) not in queried
    }
    for child in ast.walk(node):
        if (tensor := _pointer_strategy_tensor(child)) is not None:
            tensors.add(renames.get(tensor, tensor))
    return frozenset(tensors)


def _inexact_repeated_loop(
    loop: _Statement,
    statements: list[_Statement],
    scope: LaneScope,
    attached: list[_Statement],
    renames: Mapping[str, str],
    surrounding: Sequence[tuple[int, _Statement, bool]],
) -> str | None:
    """Why repeating ``loop``, whose values change with ``scope``'s lane, once per lane is observable inside it.

    A loop of the body (a device loop, or a ``while`` loop) that reads the
    lane's values, or that a barrier or a call of unknown effects pins, runs
    whole in every iteration of the lane loop: the second lane's pass begins
    after the first pass ran every iteration of it.  A statement of its body
    whose values ignore the lane -- transitively, from the lane's coordinates
    through the setup, the body's definitions and the loop's own -- is
    repeated by every pass, and unlike a statement repeated on its own its
    repetitions are separated by every iteration of the loop.  A load of a
    tensor the loop stores to reads, in the second pass, what the first
    pass's iterations stored, those after it as much as those before it; a
    store to a tensor the loop loads, or stores per lane, lands after the
    first pass's accesses of it, which the program orders the other way
    round; and a load and a store of one tensor in one statement accumulate
    once per pass.  Two lane-invariant stores re-apply the same values in
    the same order.  Registers are the placement's business when
    the loop's accumulators are initialized within the pass; one
    carried in from outside the lane loop (an enclosing device loop's
    carry, read before the pass assigns it) is updated once per pass
    (``_carried_writes``).  A statement of unknown effects whose
    operands ignore the lane may read or write its tensors in every
    pass: it meets any other access of them in the loop.  Each pass
    also runs between the statements around the loop (``surrounding``: each
    with the loop's position in its numbering and whether it varies with the
    lane), and a lane-invariant statement of the body is checked against them
    like a lane-invariant statement at the loop's place
    (``_repetition_conflict``; a load of ``x[r, c0]`` in the loop re-reads,
    in the second pass, what the first pass's per-lane store after the loop
    wrote), on the tensors it addresses through their pointers
    (``_pointer_tensors``): an array an emitter builds and indexes itself (a
    warp-MMA accumulator fragment zeroed and read out around its K loop) is
    its scratch, filled again within each pass.  The per-lane accesses are
    the barrier pass's business, which pairs them through their address
    mappings (``add_thread_barriers``): the check here knows the names an
    access reads, not whether they change its element.
    """
    lane_var = scope.lane_var
    inside = [
        _analyze(index, node, renames)
        for index, node in enumerate(_simple_statements(loop.node))
    ]
    around = [
        _analyze(-1, node, renames)
        for statement in statements
        if statement is not loop
        for node in _simple_statements(statement.node)
    ]
    setup = [_analyze(-1, node, renames) for node in scope.setup]
    everything = [*around, *setup, *attached, *inside]
    seeds = [
        {lane_var} if statement.reads & scope.coordinates else set()
        for statement in everything
    ]
    dependences = _propagate_dependence(everything, seeds, scope.masks)
    varying = [lane_var in lanes for lanes in dependences[-len(inside) :]]
    # A variable the pass reads before assigning it is carried in from
    # outside the lane loop, where no pass initializes it again.
    prefix = sorted(
        (other for _, other, _ in surrounding if 0 <= other.index < loop.index),
        key=operator.attrgetter("index"),
    )
    carried = _carried_writes(
        [*(other.node for other in prefix), *(statement.node for statement in inside)],
        renames,
    )[len(prefix) :]
    for statement, varies, updated in zip(inside, varying, carried, strict=True):
        if varies:
            continue
        if statement.pinned:
            # A call of unknown effects whose operands ignore the lane runs
            # again in every pass, between every iteration's accesses of its
            # tensors and the accesses around the loop: it may read what a
            # later iteration of the first pass stored, or store over it.
            for other_varies, shared in [
                *(
                    (other_varies, statement.tensors & other.tensors)
                    for other, other_varies in zip(inside, varying, strict=True)
                    if other is not statement
                ),
                *(
                    (
                        other_varies,
                        statement.tensors
                        & other.tensors
                        & _pointer_tensors(statement.node, renames),
                    )
                    for _, other, other_varies in surrounding
                ),
            ]:
                if not shared:
                    continue
                tensor = min(shared)
                return (
                    f"a call of unknown effects on {tensor} in a loop repeated "
                    f"whole once per {lane_var} meets "
                    f"a{' per-lane' if other_varies else 'nother'} access of it "
                    "across the passes"
                )
            continue
        if statement.atomic:
            return _repeated_atomic(statement, lane_var)
        if statement.tensors_read & statement.tensors_written:
            return (
                f"{ast.unparse(statement.node)} depends on its own result and "
                f"would repeat once per {lane_var}"
            )
        if updated := updated - statement.subscripted:
            return (
                f"{_repeated_carry(statement, updated, lane_var)} from a loop "
                "repeated whole"
            )
        for other, other_varies in zip(inside, varying, strict=True):
            if other is statement:
                continue
            for tensor in sorted(statement.tensors & other.tensors):
                writes = tensor in statement.tensors_written
                other_writes = other.pinned or tensor in other.tensors_written
                other_reads = other.pinned or tensor in other.tensors_read
                if not (writes or other_writes):
                    continue
                if writes and other_writes and not (other_reads or other_varies):
                    # Two lane-invariant stores, re-applied in program order.
                    continue
                other_access = (
                    "atomic on"
                    if other.atomic
                    else "store to"
                    if other_writes
                    else "load of"
                )
                return (
                    f"a lane-invariant {'store to' if writes else 'load of'} "
                    f"{tensor} in a loop repeated whole once per {lane_var} "
                    f"meets a{' per-lane' if other_varies else 'nother'} "
                    f"{other_access} it across the passes"
                )
        pointed = _pointer_tensors(statement.node, renames)
        for position, other, other_varies in surrounding:
            shared = statement.tensors & other.tensors & pointed
            if not shared:
                continue
            reason = _repetition_conflict(
                dataclasses.replace(statement, index=position, sites=()),
                other,
                other_varies,
                shared,
            )
            if reason is not None:
                return f"{reason} once per {lane_var} from a loop repeated whole"
    return None


def _branch_items(
    node: ast.If, guards: frozenset[str] = frozenset()
) -> list[tuple[ast.AST, frozenset[str]]]:
    """The statements of ``node``'s sides in program order, nested branches flattened.

    Each comes with the conjuncts of the branches around it inside ``node``,
    ``guards`` included (``_guarded_statements``); a loop is one statement.
    """
    items: list[tuple[ast.AST, frozenset[str]]] = []
    for children, inside in (
        (node.body, guards | _conjuncts(node.test)),
        (node.orelse, guards),
    ):
        for child in children:
            if isinstance(child, ast.If):
                items.extend(_branch_items(child, inside))
            else:
                items.append((child, inside))
    return items


def _inexact_repeated_branch(
    branch: _Statement,
    executed: list[tuple[_Statement, bool]],
    statements: list[_Statement],
    scope: LaneScope,
    attached: list[_Statement],
    renames: Mapping[str, str],
) -> str | None:
    """Why repeating ``branch``, whose values change with ``scope``'s lane, once per lane is observable inside it.

    A branch is emitted whole inside the lane loop: its statements are not
    distributed, so every lane's pass runs each statement of its sides.  One
    whose values ignore the lane -- transitively, from the lane's
    coordinates through the setup and the definitions around and inside the
    branch -- is repeated once per lane between the per-lane statements of
    the passes, inside the branch as outside it, exactly like a
    lane-invariant statement of the loop's own list (``_inexact_nest``): a
    load of ``x[r, c0]`` before a per-lane store to ``x[r, c]`` re-reads, in
    the second lane's pass, the element the first pass stored.  The sides'
    statements (nested branches flattened, a device loop kept whole and
    checked as one, ``_inexact_repeated_loop``) are checked in program order
    against each other and, from the branch's position, against the loop's
    other statements; a side some lanes skip only drops repetitions, and a
    statement guarded by the loop's first-lane predicate (an atomic
    ``_pin_repeated_atomics`` pinned) runs once.
    """
    lane_var = scope.lane_var
    items = _branch_items(cast("ast.If", branch.node))
    inside = [
        _analyze(position, node, renames)
        for position, (node, _guards) in enumerate(items)
    ]
    first_lane = (
        _conjuncts(cast("ast.expr", expr_from_string(scope.first_lane)))
        if scope.first_lane
        else None
    )
    around = [
        _analyze(-1, node, renames)
        for statement in statements
        if statement is not branch
        for node in _simple_statements(statement.node)
    ]
    setup = [_analyze(-1, node, renames) for node in scope.setup]
    everything = [*around, *setup, *attached, *inside]
    seeds = [
        {lane_var} if statement.reads & scope.coordinates else set()
        for statement in everything
    ]
    dependences = _propagate_dependence(everything, seeds, scope.masks)
    varying = [
        statement.pinned or lane_var in lanes
        for statement, lanes in zip(inside, dependences[-len(inside) :], strict=True)
    ]
    for statement, varies, (_node, guards) in zip(inside, varying, items, strict=True):
        if varies:
            if isinstance(statement.node, (ast.For, ast.While)):
                context = [
                    *(other for other in statements if other is not branch),
                    *(other for other in inside if other is not statement),
                ]
                surrounding = [
                    *(
                        (statement.index, other, other_varies)
                        for other, other_varies in zip(inside, varying, strict=True)
                        if other is not statement
                    ),
                    *(
                        (branch.index, other, other_varies)
                        for other, other_varies in executed
                        if other is not branch
                    ),
                ]
                reason = _inexact_repeated_loop(
                    statement, context, scope, attached, renames, surrounding
                )
                if reason is not None:
                    return reason
            continue
        if first_lane is not None and first_lane <= guards:
            # Issued at the loop's first lane only.
            continue
        if statement.atomic:
            return _repeated_atomic(statement, lane_var)
        if (statement.reads & statement.writes) - statement.subscripted or (
            statement.tensors_read & statement.tensors_written
        ):
            return (
                f"{ast.unparse(statement.node)} depends on its own result and "
                f"would repeat once per {lane_var} inside a branch"
            )
        # Against the loop's other statements the branch's position orders it.
        at_branch = dataclasses.replace(statement, index=branch.index)
        pairs = [
            *(
                (statement, other, other_varies)
                for other, other_varies in zip(inside, varying, strict=True)
                if other is not statement
            ),
            *(
                (at_branch, other, other_varies)
                for other, other_varies in executed
                if other is not branch
            ),
        ]
        for repeated, other, other_varies in pairs:
            shared = repeated.tensors & other.tensors
            if not shared:
                continue
            reason = _repetition_conflict(repeated, other, other_varies, shared)
            if reason is not None:
                return f"{reason} once per {lane_var} inside a branch"
    return None


def _inexact_nest(
    statements: list[_Statement],
    scopes: list[LaneScope],
    attached: dict[str, list[_Statement]],
    *,
    full_nest: bool,
    renames: Mapping[str, str],
) -> str | None:
    """Why the emitted lane loop nest computes something other than the program.

    Each loop repeats every statement it runs whose values do not depend on
    it (``dataflow_lanes``) once per iteration: a body statement placed inside
    it (every statement when the full nest is kept) that reads only the loop's
    mask or that the loop of a packet load it reads dragged in, an attached
    statement of an inner loop nested in the loop's instance that reads none
    of the loop's values.  A loop of the body (a device loop or a ``while``
    loop) whose values depend on the lane loop, or that a barrier or a call
    of unknown effects inside it pins, still repeats the accesses inside it
    that do not, with every iteration of it between two repetitions and
    between the loop list's statements around it (``_inexact_repeated_loop``);
    so does a branch whose values depend on the lane loop, with its sides'
    statements (``_inexact_repeated_branch``).  Any other statement
    with unknown effects stays where the structure pins it and is not
    checked for repetition; it still orders the checked statements around
    it.  Run for the nest the caller emits: the original one, or the
    placement, in which a statement runs inside the loops of ``lanes`` and an
    inner loop's instance nests inside an outer loop's when a statement runs
    in both.
    """
    for depth, scope in enumerate(scopes):
        lane_var = scope.lane_var
        inside = [s for s in statements if full_nest or lane_var in s.lanes]
        inner = {other.lane_var for other in scopes[depth + 1 :]}
        nested = inner if full_nest else _used_inner(inside, inner)
        executed = [(s, s.pinned or lane_var in s.dataflow_lanes) for s in inside]
        executed.extend((s, True) for s in attached[lane_var])
        body = [s for s in inside if s.index >= 0]
        carried = dict(
            zip(
                (id(s) for s in body),
                _carried_writes([s.node for s in body], renames),
                strict=True,
            )
        )
        for other in scopes[depth + 1 :]:
            if other.lane_var in nested:
                executed.extend(
                    (s, s.pinned or lane_var in s.dataflow_lanes)
                    for s in attached[other.lane_var]
                )
        for statement, varies in executed:
            if varies:
                if statement.index >= 0 and isinstance(
                    statement.node, (ast.For, ast.While)
                ):
                    reason = _inexact_repeated_loop(
                        statement,
                        statements,
                        scope,
                        attached[lane_var],
                        renames,
                        [
                            (statement.index, other, other_varies)
                            for other, other_varies in executed
                            if other is not statement
                        ],
                    )
                    if reason is not None:
                        return reason
                elif statement.index >= 0 and isinstance(statement.node, ast.If):
                    reason = _inexact_repeated_branch(
                        statement,
                        executed,
                        statements,
                        scope,
                        attached[lane_var],
                        renames,
                    )
                    if reason is not None:
                        return reason
                continue
            if statement.atomic:
                return _repeated_atomic(statement, lane_var)
            if (statement.exposed_reads & statement.writes) - statement.subscripted or (
                statement.tensors_read & statement.tensors_written
            ):
                return (
                    f"{ast.unparse(statement.node)} depends on its own result and "
                    f"would repeat once per {lane_var}"
                )
            if updated := carried.get(id(statement), frozenset()) - (
                statement.subscripted
            ):
                return _repeated_carry(statement, updated, lane_var)
            for other, other_varies in executed:
                if other is statement:
                    continue
                shared = statement.tensors & other.tensors
                if not shared:
                    continue
                reason = _repetition_conflict(statement, other, other_varies, shared)
                if reason is not None:
                    return f"{reason} once per {lane_var}"
    return None


def _require_exact_nest(
    statements: list[_Statement],
    scopes: list[LaneScope],
    attached: dict[str, list[_Statement]],
    *,
    full_nest: bool,
    renames: Mapping[str, str],
) -> None:
    reason = _inexact_nest(
        statements, scopes, attached, full_nest=full_nest, renames=renames
    )
    if reason is not None:
        raise inexact_nest_error(reason)


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
            masks=canonical(scope.masks),
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
    _propagate_dataflow_lanes(
        statements,
        scopes,
        {
            scope.lane_var: [
                *(_analyze(-1, node, rename_groups) for node in scope.setup),
                *attached[scope.lane_var],
            ]
            for scope in scopes
        },
    )
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
    exact.  Whichever nest is emitted is checked: it raises
    ``BackendUnsupported`` when a loop would repeat a memory access whose
    values ignore it around per-lane accesses of the same tensor, or an
    atomic sits in a loop its values ignore without being uniform along it
    (see the module docstring).  An atomic pinned to a loop's first lane
    replaces its statement in ``body``.  A body holding a ``_helion_*``
    placeholder is left to the pass expanding it, only its atomics pinned
    (``_pin_atomics_beside_compiler_markers``).
    """
    if not body or not scopes:
        return None
    if markers := _structural_markers(body):
        _pin_atomics_beside_compiler_markers(body, scopes, markers)
        return None
    statements, scopes, attached = _prepare(body, scopes, rename_groups)
    placement = None
    if _propagate_lanes(statements, scopes, attached):
        _pin_repeated_atomics(
            body, statements, scopes, attached, rename_groups, full_nest=False
        )
        _attribute_sites(statements, scopes, attached)
        placement = _place(statements, list(scopes), attached)
    else:
        log.debug(
            "lane placement abandoned: a loop would have to nest inside a loop "
            "the original nest places inside it (%s)",
            [scope.lane_var for scope in scopes],
        )
    if placement is None:
        # The original nest is emitted.  A pin applied for the abandoned
        # placement stays (``pinned_lanes``); the loops it left are pinned or
        # rejected for the full nest.
        _pin_repeated_atomics(
            body, statements, scopes, attached, rename_groups, full_nest=True
        )
        _attribute_sites(statements, scopes, attached)
        _require_exact_nest(
            statements, scopes, attached, full_nest=True, renames=rename_groups
        )
        return None
    # A placement is not exact by construction: an inner loop nested in an
    # outer one by ``requires`` runs its statements inside the outer loop
    # whether or not their values change with it.
    _require_exact_nest(
        statements, scopes, attached, full_nest=False, renames=rename_groups
    )
    every = {scope.lane_var for scope in scopes}
    if all(statement.lanes == every for statement in statements):
        # The placement is the original nest, which the caller emits as is.
        return None
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
    keeps the nest.  Pins the atomics of ``body`` like that function does.
    """
    reason = full_nest_inexactness(body, scopes, rename_groups=rename_groups)
    if reason is not None:
        raise inexact_nest_error(reason)


def inexact_nest_error(reason: str) -> exc.BackendUnsupported:
    """The error for a lane loop nest that does not run the tile program."""
    return exc.BackendUnsupported(
        "cute", f"the lane loop nest is not the tile program: {reason}"
    )


def full_nest_inexactness(
    body: list[ast.AST],
    scopes: list[LaneScope],
    *,
    rename_groups: Mapping[str, str],
) -> str | None:
    """Why the full lane loop nest of ``body`` is not the tile program, or None.

    ``check_full_nest`` without the raise, for a nest built around ``body``
    before it existed (a device loop's lane loops,
    ``DeviceLoopState.check_lane_loop_nest``), which is redistributed when
    inexact.  Pins the atomics of ``body`` the same way.
    """
    if not body or not scopes:
        return None
    if markers := _structural_markers(body):
        _pin_atomics_beside_compiler_markers(body, scopes, markers)
        return None
    statements, scopes, attached = _prepare(body, scopes, rename_groups)
    _propagate_lanes(statements, scopes, attached)
    _pin_repeated_atomics(
        body, statements, scopes, attached, rename_groups, full_nest=True
    )
    _attribute_sites(statements, scopes, attached)
    return _inexact_nest(
        statements, scopes, attached, full_nest=True, renames=rename_groups
    )


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


def statements_inside_loop(
    placement: list[ast.AST | LanePlacement], lane_var: str
) -> list[ast.AST]:
    """The statements ``placement`` emits inside the loop of ``lane_var``, in order.

    Statements of the loops nested in that instance are included: they run
    inside each of its iterations.  Empty when the loop is not materialized.
    """
    inside: list[ast.AST] = []

    def walk(items: list[ast.AST | LanePlacement], within: bool) -> None:
        for item in items:
            if isinstance(item, LanePlacement):
                walk(item.items, within or item.lane_var == lane_var)
            elif within:
                inside.append(item)

    walk(placement, False)
    return inside
