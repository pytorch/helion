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
lane-invariant neighbours may still leave the loops around it.

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
Such a body is rejected (``BackendUnsupported``) rather than compiled into a
nest that computes something other than the program, and an atomic placed in
a loop its values ignore is pinned or rejected on every path.  A placement is
not exact by construction: the statements of an inner loop nested inside an
outer one by its hoisted packets run inside the outer loop whether or not
their values change with it, so a body that would be rejected as the full
nest is rejected as a placement too.

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
from ..ast_extension import create
from ..ast_extension import expr_from_string
from ..ast_read_writes import HELION_ATOMIC_UNIFORM_LANES_ATTR
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
    # The per-lane index and mask definitions materialized at the top of the
    # loop's body.  Like the attached statements they are never placed; the
    # lanes their reads carry flow into the statements reading their results
    # (``_propagate_dataflow_lanes``).
    setup: tuple[ast.AST, ...] = ()
    # The predicate selecting the loop's first iteration (its lane variable
    # and constexpr V-loop variable both zero), which pins an atomic that is
    # uniform along the loop's tile axis (``_pin_repeated_atomics``).
    first_lane: str = ""
    # The tile masks among the setup's definitions.  Reading one places a
    # statement inside the loop without making its values depend on it.
    masks: frozenset[str] = frozenset()


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
# The persistent-reduction markers (``cute/persistent_branch_vec.py``) a later
# pass expands into vector accesses: a load through the pointer in its fifth
# argument and a store through the pointer in its fourth.  Bodies holding
# them never reach ``distribute_lane_loops``; the lane split's tail ordering
# (``tile_strategy._lane_invariant_tail_after_consume``) analyzes them here.
_PERSISTENT_VEC_LOAD = "_helion_persistent_branch_vec_load"
_PERSISTENT_VEC_STORE = "_helion_persistent_branch_vec_store"
_PERSISTENT_VEC_STORE_ADDRESS = 3
_RANGE_CALLS = frozenset({"range", "cutlass.range", "cutlass.range_constexpr"})
# Float max / min emulated with integer atomics (``cute/atomic_helpers.py``).
_ATOMIC_HELPERS = frozenset({"_cute_atomic_max_float32", "_cute_atomic_min_float32"})
# The vector type argument of a 16-byte ``cute.arch.load``, the store
# protocol's register-only conversion of a whole byte packet and the layout
# arithmetic of an atomic's address.
_PURE_CALLS = frozenset(
    {"ir.VectorType.get", "_cute_signed_bitfield_to_bf16_packed", "cute.crd2idx"}
)


def contains_compiler_marker(body: list[ast.AST]) -> bool:
    """Whether a later pass still has to expand a ``_helion_*`` placeholder."""
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id.startswith("_helion_")
        for statement in body
        for node in ast.walk(statement)
    )


def _is_persistent_vec_access(call: ast.Call, marker: str) -> bool:
    return (
        isinstance(call.func, ast.Name)
        and call.func.id == marker
        and len(call.args) == 6
        and not call.keywords
    )


def _is_plain_store(call: ast.Call) -> bool:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr in ("store", "__setitem__")
    return (
        isinstance(call.func, ast.Name) and call.func.id in _PLAIN_STORE_HELPERS
    ) or _is_persistent_vec_access(call, _PERSISTENT_VEC_STORE)


def _is_plain_load(call: ast.Call) -> bool:
    return (
        isinstance(call.func, ast.Name) and call.func.id in _PLAIN_LOAD_HELPERS
    ) or _is_persistent_vec_access(call, _PERSISTENT_VEC_LOAD)


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


def _is_atomic_call(call: ast.Call) -> bool:
    """``cute.arch.atomic_*(ptr, ...)`` or one of the float max / min helpers."""
    if isinstance(call.func, ast.Attribute):
        return call.func.attr.startswith("atomic_") and (
            ast.unparse(call.func.value) == "cute.arch"
        )
    return isinstance(call.func, ast.Name) and call.func.id in _ATOMIC_HELPERS


def _uniform_lanes_of(call: ast.Call) -> frozenset[str]:
    """The lane loops the atomic call is recorded as uniform along.

    ``cute/atomic_ops.py`` records them on the calls it guards; an atomic
    another lowering emits carries no record and is uniform along no loop.
    """
    return frozenset(getattr(call, HELION_ATOMIC_UNIFORM_LANES_ATTR, ()))


def _is_relaxed_atomic(call: ast.Call) -> bool:
    return any(
        keyword.arg == "sem"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value == "relaxed"
        for keyword in call.keywords
    )


def _atomic_addresses_a_tensor(call: ast.Call) -> bool:
    """Whether the atomic's pointer operand (its first argument) names its tensor."""
    return bool(call.args) and bool(_addressed_tensors(call.args[0]))


def _calls_are_movable(node: ast.AST, *, atomics: bool = False) -> bool:
    return all(
        _is_plain_store(call)
        or _is_plain_load(call)
        or _is_list_append(call)
        or (atomics and _is_atomic_call(call))
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


def _is_movable(node: ast.AST, *, atomics: bool = False) -> bool:
    """Whether running ``node`` once instead of once per lane preserves it.

    Assignments and expression statements made of proven pure calls (loads
    included), ordinary stores and store buffer appends qualify, as do
    ``range`` loops and branches built only from them.  Anything else pins
    the statement.  With ``atomics`` the atomic calls qualify too: such a
    statement is placed by its reads and its tensor and never repeated.
    """
    if isinstance(node, ast.For):
        return (
            not node.orelse
            and _is_movable_iterator(node.iter)
            and all(_is_movable(child, atomics=atomics) for child in node.body)
        )
    if isinstance(node, ast.If):
        return _calls_are_movable(node.test) and all(
            _is_movable(child, atomics=atomics) for child in [*node.body, *node.orelse]
        )
    if isinstance(node, ast.Pass):
        return True
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr)):
        return _calls_are_movable(node, atomics=atomics)
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
    if _is_persistent_vec_access(call, _PERSISTENT_VEC_STORE):
        return call.args[_PERSISTENT_VEC_STORE_ADDRESS]
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
    # An atomic read-modify-write of the tensors its calls address: placed
    # like a store, never repeated.  A ``fence`` (an atomic that is not
    # relaxed) keeps every memory access on its side of it.
    atomic: bool
    fence: bool
    # The lane loops the statement is placed in (``_propagate_lanes``): the
    # loops whose values it reads, the loops those loops nest inside and, for
    # a name it writes, the loops of every other definition of that name.
    lanes: set[str] = dataclasses.field(default_factory=set)
    # The lane loops whose iteration changes the statement's values
    # (``_propagate_dataflow_lanes``): the coordinates it reads, transitively
    # through the names it reads, the loops' masks excepted.  A subset of
    # ``lanes``; the loops in the difference repeat the statement once per
    # iteration.
    dataflow_lanes: set[str] = dataclasses.field(default_factory=set)
    # For an atomic statement: the lane loops along whose tile axes every
    # atomic call in it is uniform (``cute/atomic_ops.py``; a loop repeats
    # the statement unless it is here or in ``dataflow_lanes``), the loops
    # any of its calls is uniform along (run inside such a loop without
    # varying with it, the statement is pinned to the loop's first lane),
    # and the loops it has been pinned to.
    uniform_lanes: frozenset[str] = frozenset()
    partly_uniform_lanes: frozenset[str] = frozenset()
    # The names the statement loads only to address a subscript store
    # (``x[i] = v``): writes that are not a dependence on its own result.
    addressed_only: frozenset[str] = frozenset()
    pinned_lanes: set[str] = dataclasses.field(default_factory=set)
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
    atomics = [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and _is_atomic_call(call)
    ]
    atomic = (
        bool(atomics)
        and _is_movable(node, atomics=True)
        and all(_atomic_addresses_a_tensor(call) for call in atomics)
    )
    uniform_lanes: frozenset[str] = frozenset()
    partly_uniform_lanes: frozenset[str] = frozenset()
    if atomic:
        per_call = [_uniform_lanes_of(call) for call in atomics]
        uniform_lanes = frozenset.intersection(*per_call)
        partly_uniform_lanes = frozenset().union(*per_call)
    # ``x[i] = v`` loads ``x`` only to address the store and does not read
    # the stored values: the name is a write alone, not a dependence of the
    # statement on its own result (an augmented assignment reads it).
    store_bases = {
        id(target.value)
        for assign in ast.walk(node)
        if isinstance(assign, ast.Assign)
        for target in assign.targets
        if isinstance(target, ast.Subscript)
    }
    loads = [
        name
        for name in ast.walk(node)
        if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Load)
    ]
    addressed_only = {name.id for name in loads if id(name) in store_bases} - {
        name.id for name in loads if id(name) not in store_bases
    }
    return _Statement(
        index,
        node,
        canonical(rw.reads),
        canonical(rw.writes),
        tensors,
        canonical(read),
        canonical(written),
        pinned=not atomic and not _is_movable(node),
        register_only=not tensors and _is_register_only(node),
        atomic=atomic,
        fence=atomic and not all(_is_relaxed_atomic(call) for call in atomics),
        uniform_lanes=uniform_lanes,
        partly_uniform_lanes=partly_uniform_lanes,
        addressed_only=canonical(addressed_only),
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
    if (earlier.fence or later.fence) and not (
        earlier.register_only or later.register_only
    ):
        return True
    if earlier.writes & (later.reads | later.writes) or earlier.reads & later.writes:
        return True
    return bool(
        earlier.tensors_written & later.tensors
        or later.tensors_written & earlier.tensors
    )


def _conflicts(outside: _Statement, attached: _Statement) -> bool:
    """Whether moving ``outside`` past a loop's ``attached`` statement is observable."""
    if attached.pinned or (outside.fence and attached.tensors):
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

    A statement runs inside the loops whose values it reads, inside every
    loop when its effects are unknown, inside the loops of every other
    definition of a name it writes, and inside the loops that the loops it
    runs in nest inside (``LaneScope.requires``, the loops whose values a
    loop's attached statements read, and the outer loops of any loop a
    statement shares with them, since each loop is materialized once).
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
        # Each loop is materialized once, so a statement inside two loops
        # needs the inner loop's instance inside the outer one, and with it
        # every statement of the inner loop (the emitted nest is checked for
        # the repetition this adds; the alternative, the whole body in every
        # loop, repeats more).
        for statement in statements:
            nested = sorted(statement.lanes, key=order.__getitem__)
            for depth, inner in enumerate(nested):
                for outer in nested[:depth]:
                    changed = require(inner, outer) or changed
    return all(
        order[outer] < order[lane_var]
        for lane_var, outers in requires.items()
        for outer in outers
    )


def _propagate_dataflow_lanes(
    statements: list[_Statement],
    scopes: list[LaneScope],
    loop_statements: dict[str, list[_Statement]],
) -> None:
    """Assign every statement the lane loops whose iteration changes its values.

    A statement depends on a loop when it reads one of the loop's coordinates
    or a name whose definition does, transitively through the loop's setup
    (an index built from the lane variable), the statements attached to a
    vectorized loop (a packet whose load address reads an outer loop's index)
    and the body statements.  A loop's tile mask (``LaneScope.masks``) carries
    no dependence: it skips a statement in the iterations past the tile's end
    and leaves the address and value of the other iterations alone, so a
    statement reading nothing else of the loop is the same access in every
    iteration.  Unlike ``_propagate_lanes`` nothing is added for the nesting
    the structure forces or for the other definitions of a name a statement
    writes.
    """
    coordinate_lanes = {
        name: scope.lane_var for scope in scopes for name in scope.coordinates
    }
    masks = frozenset().union(*(scope.masks for scope in scopes))
    everything = [
        *statements,
        *(s for scope in scopes for s in loop_statements[scope.lane_var]),
    ]
    for statement in everything:
        statement.dataflow_lanes = {
            coordinate_lanes[name]
            for name in statement.reads
            if name in coordinate_lanes
        }
    changed = True
    while changed:
        changed = False
        name_lanes: dict[str, set[str]] = {}
        for statement in everything:
            for name in statement.writes - masks:
                name_lanes.setdefault(name, set()).update(statement.dataflow_lanes)
        for statement in everything:
            size = len(statement.dataflow_lanes)
            for name in statement.reads:
                statement.dataflow_lanes |= name_lanes.get(name, set())
            changed = changed or len(statement.dataflow_lanes) != size


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
            raise exc.BackendUnsupported(
                "cute",
                "the lane loop nest is not the tile program: "
                + _repeated_atomic(statement, min(repeated, key=order.__getitem__)),
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
                raise exc.BackendUnsupported(
                    "cute", f"the lane loop nest is not the tile program: {reason}"
                )
        predicate = " and ".join(by_lane[lane_var].first_lane for lane_var in lane_vars)
        node = _pinned(statement.node, predicate)
        if node is None:
            raise exc.BackendUnsupported("cute", _unpinnable(statement.node, lane_vars))
        varying = pinned & statement.dataflow_lanes
        if varying:
            raise exc.BackendUnsupported(
                "cute",
                "the lane loop nest is not the tile program: "
                + _varying_pin(statement, sorted(varying, key=order.__getitem__)),
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


def _inexact_nest(
    statements: list[_Statement],
    scopes: list[LaneScope],
    attached: dict[str, list[_Statement]],
    *,
    full_nest: bool,
) -> str | None:
    """Why the emitted lane loop nest computes something other than the program.

    Each loop repeats every statement it runs whose values do not depend on
    it (``dataflow_lanes``) once per iteration: a body statement placed inside
    it (every statement when the full nest is kept) that reads only the loop's
    mask or that the loop of a packet load it reads dragged in, an attached
    statement of an inner loop nested in the loop's instance that reads none
    of the loop's values.  A statement with unknown effects stays where the
    structure pins it and is not checked for repetition; it still orders the
    checked statements around it.  Run for the nest the caller emits: the
    original one, or the placement, in which a statement runs inside the
    loops of ``lanes`` and an inner loop's instance nests inside an outer
    loop's when a statement runs in both.
    """
    for depth, scope in enumerate(scopes):
        lane_var = scope.lane_var
        inside = [s for s in statements if full_nest or lane_var in s.lanes]
        inner = {other.lane_var for other in scopes[depth + 1 :]}
        nested = inner if full_nest else _used_inner(inside, inner)
        executed = [(s, s.pinned or lane_var in s.dataflow_lanes) for s in inside]
        executed.extend((s, True) for s in attached[lane_var])
        for other in scopes[depth + 1 :]:
            if other.lane_var in nested:
                executed.extend(
                    (s, s.pinned or lane_var in s.dataflow_lanes)
                    for s in attached[other.lane_var]
                )
        for statement, varies in executed:
            if varies:
                continue
            if statement.atomic:
                return _repeated_atomic(statement, lane_var)
            if (statement.reads & statement.writes) - statement.addressed_only:
                return (
                    f"{ast.unparse(statement.node)} depends on its own result and "
                    f"would repeat once per {lane_var}"
                )
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
) -> None:
    reason = _inexact_nest(statements, scopes, attached, full_nest=full_nest)
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
    replaces its statement in ``body``.
    """
    if not body or not scopes or contains_compiler_marker(body):
        return None
    statements, scopes, attached = _prepare(body, scopes, rename_groups)
    placement = None
    if _propagate_lanes(statements, scopes, attached):
        _pin_repeated_atomics(
            body, statements, scopes, attached, rename_groups, full_nest=False
        )
        _attribute_sites(statements, scopes, attached)
        placement = _place(statements, list(scopes), attached)
    if placement is None:
        # The original nest is emitted.  A pin applied for the abandoned
        # placement stays (``pinned_lanes``); the loops it left are pinned or
        # rejected for the full nest.
        _pin_repeated_atomics(
            body, statements, scopes, attached, rename_groups, full_nest=True
        )
        _attribute_sites(statements, scopes, attached)
        _require_exact_nest(statements, scopes, attached, full_nest=True)
        return None
    # A placement is not exact by construction: an inner loop nested in an
    # outer one by ``requires`` runs its statements inside the outer loop
    # whether or not their values change with it.
    _require_exact_nest(statements, scopes, attached, full_nest=False)
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
    use, and for a nest built around ``body`` before it existed (a device
    loop's lane loops, ``DeviceLoopState.check_lane_loop_nest``);
    ``distribute_lane_loops`` performs the same check itself whenever it
    keeps the nest.  Pins the atomics of ``body`` like that function does.
    """
    if not body or not scopes or contains_compiler_marker(body):
        return
    statements, scopes, attached = _prepare(body, scopes, rename_groups)
    _propagate_lanes(statements, scopes, attached)
    _pin_repeated_atomics(
        body, statements, scopes, attached, rename_groups, full_nest=True
    )
    _attribute_sites(statements, scopes, attached)
    _require_exact_nest(statements, scopes, attached, full_nest=True)


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
