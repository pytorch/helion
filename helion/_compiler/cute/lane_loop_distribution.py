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
A device loop of the body whose values depend on the lane loop runs whole
once per lane, so the accesses inside it whose values do not are repeated
with every iteration of it between two repetitions: a load of a tensor the
loop stores to, or a store to one it loads or stores per lane, is observable
there in any order, the first pass's later iterations coming before the
second pass's earlier ones.
Such a body is rejected (``BackendUnsupported``) rather than compiled into a
nest that computes something other than the program, and an atomic placed in
a loop its values ignore is pinned or rejected on every path.  A placement is
not exact by construction: the statements of an inner loop nested inside an
outer one by its hoisted packets run inside the outer loop whether or not
their values change with it, so a body that would be rejected as the full
nest is rejected as a placement too.

The placement and the exactness checks reason about one thread's program
order.  The threads of the block run the same nest unsynchronized, and a
statement whose address ignores a tile axis that several threads share (a
``tile.begin`` row read beside a per-lane update of the same tensor) touches
an element another thread owns: the owner's store and the reader's load are
in the program's order only if a block-wide barrier separates them.
``add_thread_barriers`` runs on whichever nest is emitted, for the grid body
and for a device loop's, and puts ``cute.arch.sync_threads()`` between two
accesses of one tensor that threads differing along a shared axis could
reorder, at their nearest common statement list (or between a vectorized
loop's hoisted packet loads, its V-loop and its store flushes).  Every
statement list of the nest runs on every thread with the same trip counts
(the tile masks and the atomics' leader predicates guard single accesses,
never the structure), so every thread reaches the barrier.  A store whose
address and value ignore the shared axes needs none against a later load
or an identical store: each thread's own copy of it precedes its read, and
identical stores win in any order.

That analysis reads the address mappings (``cute/address_maps.py``): two
accesses of a tensor can only meet on one thread along an axis when the
equations between their per-dimension terms, translated over the thread
coordinates, the loop variables and the uniform values by inlining the
body's definitions, force the coordinate equal (a mixed-radix argument over
the coordinates' ranges, with ``a % m`` and ``a // m`` together determining
``a``).  Reading the thread's coordinate is not owning the element:
``x[t0, t1.index + 1]`` beside ``x[t0, t1]``, ``x[n - 1 - t]`` or
``x[(t + 1) % n]`` beside ``x[t]`` name another thread's element, and so does
an address computed from a value loaded from memory (a gathered row
``x[t0, perm[t1]]`` is any thread's): such accesses count as touching every
thread's elements.  Two accesses whose
tile-program regions the memory-op codegen recorded
(``HELION_ACCESS_REGIONS_ATTR``: the slices ``out[t, :n1]`` and
``out[t, n1:]``) are apart when their intervals along some dimension do not
meet, and need no barrier.  A device loop runs its body once per iteration
on every thread with nothing between the iterations, while the program
orders all of iteration j before iteration j + 1: an access whose address
ignores the loop variable (``x[t0, 0]`` inside a loop over ``tk``) meets the
previous iteration's per-thread stores of its tensor, so every pair is
checked in that order as well and, unless a barrier of the body already
separates the two within an iteration, one closes the body.  A lane loop's
iterations are one tile statement's lanes, which the program runs together
before the next statement's: an access conflicting with a store of its
tensor across the lanes of one loop (the gather of lane i + 1 and the store
of lane i touch one element; so do the shifted read ``x[t + 1]`` of one lane
and the store ``x[t]`` of the next) cannot be ordered by any barrier and is
rejected, as the loop would have to be split; only a pair whose equal terms
force the loop's own variable equal is confined to its lane (the V-loop's
variable forced equal confines a pair to a vector lane, not to a lane of the
loop around it).  A device loop inside a lane loop runs whole once per lane,
so its body's accesses of one tensor are paired the same way, with the
device loop's iteration free on both sides: the previous row's element read
per lane beside the row's store (``x[t0.index - 1, tk]`` and ``x[t0, tk]``),
or an address every lane shares beside a per-lane one, is rejected; two
accesses confined to a lane, two that never meet (only the literal bounds
of their recorded regions count there: the loop's own tile begins differ
between the iterations the passes pair), or two stores whose addresses
ignore the lane (re-applied in program order) are not.  A statement's own
stores meet each other, across the lanes' passes as across the device
loop's iterations, when the map from the instance to the element folds
(the lane or thread and the iteration summed into one index); the terms
show the fold when they know every value in them, and a map through a
value loaded per thread or per lane (a jagged tile's rows begin at an
offset loaded per row) is not modelled.  A packet
(a vector load hoisted above a V-loop, the flush of its stores) is every one
of its elements: its term along the vectorized dimension is the per-thread
base's plus the V-loop's variable, so the packet's later elements are
compared as much as its first.

Memory dependences key on the generated tensor names, as the rest of the
lane-loop lowering does: two kernel arguments that view the same storage are
not ordered against each other.  A tensor the body builds itself (the
shared-memory view of the epilogue subtile's staging, the register array
carrying a scan's prefix) is accessed by subscript: ``smem[i] = v`` is a
plain store of it and ``smem[i] += v`` a read-modify-write.  The name also
carries the stored values to the later loads of the tensor, as ``ReadWrites``
counts an in-place write (a scan's prefix loaded back in the next lane
iteration depends on that iteration through the array); only the register
self-dependence check leaves it aside.

Names are read through the device function's rename groups.  A value carried
by a nested loop is still written under its loop-output name here (``v_3``)
and only a later pass renames it to the accumulator (``acc``); reading both
as one name keeps the accumulator's initialization in the loops its updates
depend on, once per lane.

Design note: the ownership model and its proofs
-----------------------------------------------
Every check above asks one question of two accesses to one tensor with a
write among them: can two instances of them (two threads, two lanes of one
thread, two iterations of a device loop) touch one element?  Three proofs
answer it, each failing closed, and the answer decides a barrier (between
threads), a rejection (between lanes) or a barrier closing the loop's body
(between iterations).

* Ownership, from the address mappings (``cute/address_maps.py``).  Each
  address term is translated over the thread coordinates, the lane and
  V-loop variables, the device loops' iterations (``start + step *
  iteration`` from their headers) and the uniform values (a kernel
  argument, a tile offset, a shape query) by inlining the definitions
  reaching the body; a value the translation cannot follow (a loaded
  index, a name a call defines or the body defines twice) is an unknown
  integer of its own, unless a definition around the body loads it at a
  uniform address from a kernel argument nothing in view writes (not from
  a tensor made in view: a thread's register array is the thread's own):
  every thread and lane then reads that element alike before the body
  runs (a segment's start by the block's index) and it is a uniform value
  (``AddressMaps.uniform_load``).  The equations "a term of the first
  access equals the same term of the second" are solved by a mixed-radix
  argument over the variables' ranges (the thread counts, the trip counts,
  the literal loop bounds, the conjuncts ``tid < c`` and ``index < end``
  of the branches around both accesses, the latter keeping the index one
  bounded quantity when the end is a literal and making the end a radix
  of the argument when it is a uniform value (``K * row + column`` under
  ``column < K`` is one row's, whatever ``K``), each read where its branch
  stands: one around a device loop knows nothing of the loop's own
  index, one inside it around every access of a tensor bounds that
  tensor's indices for the loop as a whole; a conjunct's names are read
  as the definitions they came from left them, so a mask the nest
  defines over an index the loop redefines bounds the outer index, not
  the loop's), and the variables the
  equations force equal say what the two instances share: a thread
  coordinate (one thread's, in program order), the lane variable (one
  lane's), the iteration (one iteration's).  A packet is every element of
  its V-loop; a store-only statement is paired with itself only through
  terms that know every value.  Whatever is not forced equal counts as
  another instance's.
* Regions, from the tile program (``cute/access_regions.py``).  The
  memory-op codegen records the interval each access covers per dimension
  (``HELION_ACCESS_REGIONS_ATTR``: the slices ``out[t, :n1]`` and
  ``out[t, n1:]``); two accesses apart along some dimension never meet.
  ``_disjoint`` cancels the tile begins two statements of one body share
  (``{}``), shifts a device loop's begins by its step for a later
  iteration (``loop_shifts``), and counts literal bounds only inside a
  loop the lanes repeat whole (None), whose own begins differ between the
  iterations two passes pair.
* Order, from the accesses themselves (``_races``): two loads, two
  atomics, or a store whose address and value ignore every shared axis
  before a load or another such store need no ordering whatever the
  instances.  A metadata query of a tensor or of a slice of it
  (``t[...].shape``, ``t[...].layout``, ``t.iterator.alignment``) is no
  access at all, whatever call it feeds (``_METADATA_ATTRS``).

Known conservative cases (right programs rejected or barriered): an index
bounded by an end that is an expression (``column < ends - starts``) or
offset by a grid coordinate (a row tile ``tile_offset + tid < m`` whose
block exceeds ``m``) is not a bounded quantity, and one bounded by a
uniform value ``K`` is a radix for ``K * row + column`` alone (not for a
stride ``K - 8`` or a scaled column), so a flattened store of either
kind, or a column-major one over such a padded row tile, is rejected
under the row lanes and closes the body under row threads; a
masked load's condition (``p.load() if mask else 0``) is no branch, so a
mask bounds a store, not the load beside it; a load in a branch's test, a
while's test or a loop's header of a device loop the lanes repeat whole,
beside a store to its tensor, is rejected rather than paired across the
lanes; a packet the pass does not
recognize (a scalar ``cute.arch.load``, a load inside the V-loop, a
reduction's flush) is an access of unknown elements; regions with
symbolic bounds exempt no pair inside a repeated device loop;
``tile.end``, ``tile.count`` and tensor indices record no region; a pair
already ordered by a barrier closing an inner loop's body may get another
at the enclosing loop; a branch on a literal condition (an emitter's
``if True:`` scope) is divergent like any other, so a pair racing inside
it is rejected rather than barriered.  Known gaps (programs accepted
whose nest is not the program): a store through a value loaded per thread
or per lane (``x[base[t0] + tk.begin]``, a jagged tile's rows beginning at
an offset loaded per row) is not paired with itself, since the terms do
not model it and rejecting it would reject every jagged store (a base
every thread and lane loads alike, a segment's start by the block's
index, is modelled: ``AddressMaps.uniform_load``); a negative literal
index, which the CuTe codegen emits as the element before the row, is
modelled as emitted (a frontend issue outside this pass).
"""

from __future__ import annotations

import ast
import collections
from collections.abc import Mapping
import dataclasses
import logging
import operator
from typing import TYPE_CHECKING
from typing import TypeVar

import sympy

from ... import exc
from ..ast_extension import create
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..ast_read_writes import HELION_ACCESS_REGIONS_ATTR
from ..ast_read_writes import HELION_ATOMIC_UNIFORM_LANES_ATTR
from ..ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from ..ast_read_writes import ReadWrites
from ..tile_strategy import _is_proven_relocatable_call
from ..tile_strategy import _memory_write_calls
from .access_regions import access_address
from .address_maps import AddressMaps
from .address_maps import Load
from .address_maps import flat_terms
from .address_maps import iteration_symbol
from .address_maps import loop_symbol
from .address_maps import thread_symbol
from .address_maps import uniform_symbol
from .address_maps import whole_quantities
from .cache_policy_loads import _CUTE_CACHE_LOAD_HELPER_NAMES

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import Sequence

    from .address_maps import Bound

log = logging.getLogger(__name__)


_T = TypeVar("_T")


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
    # Where the constexpr V-loop sits among the vec wrapper's statements: the
    # first ``vloop_index`` attached statements precede it (the lane base, the
    # hoisted packet loads), the rest follow it (the store flushes).  Zero
    # for a plain loop, which attaches nothing.
    vloop_index: int = 0
    # The trip count of the loop and of its constexpr V-loop, by variable
    # (``address_maps.AddressMaps``: the range of a lane variable bounds the
    # elements a thread's lanes cover).
    counts: Mapping[str, int] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class LanePlacement:
    """One lane loop instance: its statements and nested loops in order."""

    lane_var: str
    items: list[ast.AST | LanePlacement]
    # Positions among the loop's vec wrapper statements (its attached
    # statements and V-loop in emission order) before which
    # ``add_thread_barriers`` puts a block-wide barrier.
    barriers: list[int] = dataclasses.field(default_factory=list)


_PLAIN_STORE_HELPERS = frozenset({"_cute_store_u16_vec", "_cute_store_u32_vec"})
# Inline-PTX cache-hinted loads (16-, 8- and 4-byte variants).
_PLAIN_LOAD_HELPERS = _CUTE_CACHE_LOAD_HELPER_NAMES
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
# Attributes querying a tensor's metadata: ``t[...].shape``, ``t[...].layout``
# and ``t.iterator.alignment`` read no element of ``t`` (an epilogue sizing
# its register fragment after a slice of the output partition).
_METADATA_ATTRS = frozenset({"shape", "layout", "stride", "element_type", "alignment"})


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


def _mentions(
    node: ast.AST, skip: Callable[[ast.AST], bool] | None = None
) -> Iterator[str]:
    """Generated tensor names addressed in ``node``, one per mention.

    ``t.iterator``, ``t.__setitem__`` and ``t[...]`` address ``t``.  A
    metadata query of the tensor or of a slice of it (``t[...].shape``,
    ``t[...].layout``, ``t.iterator.alignment``: ``_METADATA_ATTRS``) reads
    no element and addresses nothing, though an index inside the slice may
    (a load).  ``skip`` excludes subtrees.  Aliasing kernel arguments have
    distinct names and are not related here.
    """
    pending = [node]
    while pending:
        child = pending.pop()
        if skip is not None and skip(child):
            continue
        if isinstance(child, ast.Attribute) and child.attr in _METADATA_ATTRS:
            queried = child.value
            if isinstance(queried, ast.Attribute) and queried.attr == "iterator":
                queried = queried.value
            if isinstance(queried, ast.Name):
                continue
            if isinstance(queried, ast.Subscript) and isinstance(
                queried.value, ast.Name
            ):
                pending.append(queried.slice)
                continue
        if isinstance(child, ast.Attribute):
            if child.attr in ("iterator", "__setitem__") and isinstance(
                child.value, ast.Name
            ):
                yield child.value.id
        elif isinstance(child, ast.Subscript) and isinstance(child.value, ast.Name):
            yield child.value.id
        pending.extend(ast.iter_child_nodes(child))


def _tensor_mentions(node: ast.AST) -> collections.Counter[str]:
    """Generated tensor names addressed in ``node``, counted per mention (``_mentions``)."""
    return collections.Counter(_mentions(node))


def _tensor_names_outside(node: ast.AST, calls: Iterable[ast.Call]) -> set[str]:
    """Generated tensor names addressed in ``node`` outside the subtrees of ``calls``."""
    skipped = {id(call) for call in calls}
    return set(_mentions(node, lambda child: id(child) in skipped))


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
    return collections.Counter(
        _mentions(address, lambda node: isinstance(node, ast.Call))
    )


def _subscript_assignments(
    node: ast.AST,
) -> tuple[collections.Counter[str], set[str]]:
    """The tensors whose elements ``node`` assigns by subscript.

    A tensor the body builds itself (a shared-memory view of
    ``cute.make_tensor``) is accessed by subscript rather than through a
    store call: ``smem[i] = v`` is a plain store of it, counted per target so
    ``_tensor_accesses`` can discount the mention, and ``smem[i] += v`` a
    read-modify-write, returned in the second set.
    """
    plain: collections.Counter[str] = collections.Counter()
    updated: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.AugAssign):
            targets: list[ast.expr] = [child.target]
        elif isinstance(child, ast.Assign):
            targets = child.targets
        elif isinstance(child, ast.AnnAssign):
            targets = [child.target]
        else:
            continue
        for target in targets:
            for element in ast.walk(target):
                if (
                    isinstance(element, ast.Subscript)
                    and isinstance(element.ctx, ast.Store)
                    and isinstance(element.value, ast.Name)
                ):
                    if isinstance(child, ast.AugAssign):
                        updated.add(element.value.id)
                    else:
                        plain[element.value.id] += 1
    return plain, updated


def _tensor_accesses(node: ast.AST) -> tuple[set[str], set[str]]:
    """The tensor names ``node`` reads and the ones it writes.

    A plain store writes the tensor its address names, a subscript assignment
    the tensor it subscripts; every other mention (a load, a store's value, a
    read-modify-write, an augmented subscript assignment, an unknown call's
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
    plain, updated = _subscript_assignments(node)
    written |= set(plain) | updated
    read.subtract(plain)
    return {name for name, count in read.items() if count > 0}, written


def _is_register_only(node: ast.AST) -> bool:
    """Whether ``node`` computes registers from registers, without memory."""
    return isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and all(
        ast.unparse(call.func) in _PURE_CALLS
        or _is_proven_relocatable_call(call, allow_load=False)
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
    )


_Region = tuple[tuple[sympy.Expr, sympy.Expr] | None, ...]


@dataclasses.dataclass(frozen=True)
class _Access:
    """One mention of a tensor: its address and the elements it covers.

    ``element`` for one element at the address: a plain ``.load()`` /
    ``.store(v)`` through the pointer, a subscript, an atomic.  Otherwise a
    packet of consecutive elements (a vector load, a store flush) or an
    operand of a call the pass does not know: ``vector`` and ``base`` name
    the V-loop whose iterations index the packet's elements and the
    per-thread base its address reads along them, when the wrapper around
    the packet is in view (``_packets``); None when it is not.
    """

    address: ast.AST
    element: bool
    vector: str | None = None
    base: str | None = None


def _is_constexpr_loop(node: ast.AST) -> bool:
    """A V-loop: ``for v in cutlass.range_constexpr(n)`` with a literal ``n``."""
    return (
        isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and isinstance(node.iter, ast.Call)
        and ast.unparse(node.iter.func) == "cutlass.range_constexpr"
        and len(node.iter.args) == 1
        and not node.iter.keywords
        and isinstance(node.iter.args[0], ast.Constant)
        and isinstance(node.iter.args[0].value, int)
    )


def _is_device_loop(node: ast.AST) -> bool:
    """A loop of the tile program (a ``hl.tile`` / ``hl.range`` nest's ``for``): not a lane loop, not a V-loop."""
    return (
        isinstance(node, ast.For)
        and getattr(node, HELION_LANE_LOOP_VAR_ATTR, None) is None
        and not _is_constexpr_loop(node)
    )


def _is_packet_call(call: ast.Call) -> bool:
    """A vector load or store of consecutive elements through its pointer operand."""
    if isinstance(call.func, ast.Name):
        return call.func.id in _PLAIN_STORE_HELPERS or call.func.id in (
            _PLAIN_LOAD_HELPERS
        )
    return ast.unparse(call.func) in ("cute.arch.load", "cute.arch.store") and any(
        isinstance(argument, ast.Call)
        and ast.unparse(argument.func) == "ir.VectorType.get"
        for argument in call.args
    )


def _packets(node: ast.AST) -> dict[int, tuple[str, str]]:
    """The packets of the vec wrappers inside ``node``: ``(V-loop variable, per-thread base)`` by call id.

    A vec wrapper is a loop whose body holds one V-loop (``_is_constexpr_loop``)
    beside the per-thread base of its lanes, the packet loads hoisted above
    the V-loop and the store flushes after it.  A packet's address indexes
    the vectorized dimension by the base, which the wrapper's body defines
    and the V-loop's body advances by its variable; a packet reading no such
    name, or more than one, is left out (its extent unknown).
    """
    packets: dict[int, tuple[str, str]] = {}
    for loop in ast.walk(node):
        if not isinstance(loop, ast.For):
            continue
        vloops = [statement for statement in loop.body if _is_constexpr_loop(statement)]
        if len(vloops) != 1:
            continue
        (vloop,) = vloops
        assert isinstance(vloop, ast.For) and isinstance(vloop.target, ast.Name)
        around = [statement for statement in loop.body if statement is not vloop]
        defined = {
            target.id
            for statement in around
            if isinstance(statement, ast.Assign)
            for target in statement.targets
            if isinstance(target, ast.Name)
        }
        for statement in around:
            for call in ast.walk(statement):
                if not (isinstance(call, ast.Call) and _is_packet_call(call)):
                    continue
                address = access_address(call)
                if address is None:
                    continue
                bases = _names_read(address, {}) & defined
                if len(bases) == 1:
                    packets[id(call)] = (vloop.target.id, bases.pop())
    return packets


def _scope_packet(scope: LaneScope) -> tuple[str, str] | None:
    """A vectorized scope's ``(V-loop variable, per-thread base)``, the coordinates beside its lane variable."""
    vectors = [name for name in scope.counts if name != scope.lane_var]
    bases = [
        name
        for name in scope.coordinates
        if name != scope.lane_var and name not in scope.counts
    ]
    if len(vectors) == 1 and len(bases) == 1:
        return vectors[0], bases[0]
    return None


def _access_addresses(
    node: ast.AST, packets: Mapping[int, tuple[str, str]] | None = None
) -> dict[str, list[_Access]]:
    """The address of every tensor mention of ``node``, per tensor name.

    A ``t.iterator`` mention's address is the pointer arithmetic around it
    (``t.iterator + Int32(i) * Int32(t.layout.stride[0])``), a
    ``t.__setitem__(index, value)`` mention's the index and a ``t[index]``
    mention's the index, so every mention ``_tensor_mentions`` counts has one
    address here, with the elements it covers (``_Access``; ``packets`` are
    the wrappers' packets in view, ``_packets``).
    """
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(node):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    addresses: dict[str, list[_Access]] = collections.defaultdict(list)
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
            if child.attr == "iterator":
                address: ast.AST = child
                while isinstance(parents.get(address), (ast.BinOp, ast.UnaryOp)):
                    address = parents[address]
                owner = parents.get(address)
                if isinstance(owner, ast.IfExp):
                    if owner.orelse is address and child.value.id in (
                        _addressed_tensors(owner.body)
                    ):
                        # The fallback pointer of a masked load (``p if mask
                        # else p0``): a valid address of the tensor whose
                        # loaded value the select discards, not an access of
                        # its own.
                        continue
                    owner = parents.get(owner)
                addresses[child.value.id].append(
                    _pointer_access(address, owner, parents, packets or {})
                )
            elif child.attr == "__setitem__":
                call = parents.get(child)
                if isinstance(call, ast.Call) and call.func is child and call.args:
                    addresses[child.value.id].append(_Access(call.args[0], True))
                else:
                    addresses[child.value.id].append(_Access(child, True))
        elif isinstance(child, ast.Subscript) and isinstance(child.value, ast.Name):
            addresses[child.value.id].append(_Access(child.slice, True))
    return addresses


def _pointer_access(
    address: ast.AST,
    owner: ast.AST | None,
    parents: Mapping[ast.AST, ast.AST],
    packets: Mapping[int, tuple[str, str]],
) -> _Access:
    """The access through the pointer ``address``, whose consumer is ``owner``."""
    if isinstance(owner, ast.Attribute):
        call = parents.get(owner)
        if (
            isinstance(call, ast.Call)
            and call.func is owner
            and owner.attr in ("load", "store")
        ):
            return _Access(address, True)
        if isinstance(call, ast.Call) and _is_atomic_call(call):
            # ``cute.arch.atomic_add(p.llvm_ptr, ...)``: one element.
            return _Access(address, True)
        return _Access(address, False)
    if isinstance(owner, ast.Call):
        if _is_atomic_call(owner):
            return _Access(address, True)
        if _is_packet_call(owner):
            vector, base = packets.get(id(owner), (None, None))
            return _Access(address, False, vector, base)
    return _Access(address, False)


def _names_read(node: ast.AST, renames: Mapping[str, str]) -> set[str]:
    return {
        renames.get(child.id, child.id)
        for child in ast.walk(node)
        if isinstance(child, ast.Name)
    }


def _summands(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return [*_summands(node.left), *_summands(node.right)]
    return [node]


def _address_terms(address: ast.AST, tensor: str) -> list[ast.AST]:
    """The per-dimension index terms of an address of ``tensor``.

    The pointer arithmetic ``t.iterator + Int32(i) * Int32(t.layout.stride[0])
    + ...`` yields one term per dimension, as does
    ``t.iterator + cute.crd2idx((i, j), t.layout)`` (an atomic's address) and
    a subscript's tuple; any other shape is one term, the address itself.
    """
    if isinstance(address, ast.Tuple):
        return list(address.elts)
    summands = _summands(address)
    if not (
        len(summands) >= 2
        and isinstance(summands[0], ast.Attribute)
        and summands[0].attr == "iterator"
    ):
        return [address]
    if (
        len(summands) == 2
        and isinstance(summands[1], ast.Call)
        and ast.unparse(summands[1].func) == "cute.crd2idx"
        and len(summands[1].args) == 2
        and ast.unparse(summands[1].args[1]) == f"{tensor}.layout"
    ):
        coordinates = summands[1].args[0]
        return (
            list(coordinates.elts)
            if isinstance(coordinates, ast.Tuple)
            else [coordinates]
        )
    terms: list[ast.AST] = []
    for summand in summands[1:]:
        if not (
            isinstance(summand, ast.BinOp)
            and isinstance(summand.op, ast.Mult)
            and isinstance(summand.left, ast.Call)
            and len(summand.left.args) == 1
            and isinstance(summand.right, ast.Call)
            and len(summand.right.args) == 1
            and isinstance(summand.right.args[0], ast.Subscript)
            and ast.unparse(summand.right.args[0].value) == f"{tensor}.layout.stride"
        ):
            return [address]
        terms.append(summand.left.args[0])
    return terms


def _defining_load(
    statement: ast.AST, renames: Mapping[str, str]
) -> tuple[str, Load] | None:
    """``(name, load)`` when ``statement`` defines ``name`` by one plain load, masked or not.

    ``name = (t.iterator + ...).load()``, or that load selected by a mask
    (``... if mask else fill``), of one tensor mentioned nowhere else in
    the statement (``AddressMaps.uniform_load``).
    """
    if not (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    ):
        return None
    value = statement.value
    load = value.body if isinstance(value, ast.IfExp) else value
    guards = (value.test, value.orelse) if isinstance(value, ast.IfExp) else ()
    if not (
        isinstance(load, ast.Call)
        and isinstance(load.func, ast.Attribute)
        and load.func.attr == "load"
        and not load.args
        and not load.keywords
    ):
        return None
    accesses = _access_addresses(statement)
    if len(accesses) != 1:
        return None
    ((tensor, mentions),) = accesses.items()
    if len(mentions) != 1 or not mentions[0].element:
        return None
    name = statement.targets[0].id
    return renames.get(name, name), Load(
        renames.get(tensor, tensor),
        tuple(_address_terms(mentions[0].address, tensor)),
        guards,
        value,
        statement,
    )


def _access_regions(node: ast.AST) -> dict[str, tuple[_Region, ...]]:
    """The regions the memory-op codegen recorded on ``node``'s accesses, per tensor.

    ``language/memory_ops.py`` tags a load or store call with the elements the
    tile program's access covers, per tensor dimension
    (``HELION_ACCESS_REGIONS_ATTR``).  A tensor is reported only when every
    mention of it carries a region.
    """
    tagged: dict[str, list[_Region]] = collections.defaultdict(list)
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        regions = getattr(call, HELION_ACCESS_REGIONS_ATTR, None)
        address = access_address(call)
        if regions is None or address is None:
            continue
        for tensor in _addressed_tensors(address):
            tagged[tensor].append(regions)
    mentions = _tensor_mentions(node)
    return {
        tensor: tuple(regions)
        for tensor, regions in tagged.items()
        if len(regions) == mentions[tensor]
    }


def _precedes(end: sympy.Expr, begin: sympy.Expr) -> bool:
    """Whether ``end <= begin``: an interval ending at ``end`` lies before one beginning at ``begin``.

    Decided when the difference is a number or its sign follows from the
    symbols' assumptions (a block size is a positive integer).
    """
    difference = sympy.expand(sympy.Add(begin, sympy.Mul(-1, end)))
    return difference.is_nonnegative is True


_Shifts = Mapping[sympy.Symbol, sympy.Expr | None]


def _disjoint(
    first: tuple[_Region, ...],
    second: tuple[_Region, ...],
    *,
    shifts: _Shifts | None,
) -> bool:
    """Whether no region of ``first`` meets one of ``second``.

    Two regions are apart when, along some dimension, both are known
    intervals and one ends before the other begins.  Bounds may share a
    symbol (a tile's begin, a block size) standing for one value on every
    thread of the block.  ``shifts`` puts ``second`` in a later iteration
    of a loop: its keys are the begin symbols of the loop's blocks, each
    value the step of one iteration (None when several blocks step in turn,
    so the next begin is unrelated).  Advanced by one step, ``second`` may
    lie past ``first`` -- then it does in every later iteration when the
    step moves it no closer, its begin not decreasing in the loop's begin
    (a begin of ``c - begin`` comes back towards ``first`` by a block per
    iteration, and one step proves nothing about the next) -- or, when it
    does not read the loop's begin, before it.  ``shifts`` None knows no
    loop: no symbol cancels then.
    """
    if shifts is None:
        substitution: dict[sympy.Symbol, sympy.Expr] = {}
    else:
        substitution = {
            symbol: sympy.Dummy() if step is None else sympy.Add(symbol, step)
            for symbol, step in shifts.items()
        }

    def apart(
        a: tuple[sympy.Expr, sympy.Expr], b: tuple[sympy.Expr, sympy.Expr]
    ) -> bool:
        if shifts is None:
            if any(bound.free_symbols for bound in (*a, *b)):
                return False
            return _precedes(a[1], b[0]) or _precedes(b[1], a[0])
        advanced = b[0].xreplace(substitution)
        if _precedes(a[1], advanced) and _precedes(b[0], advanced):
            return True
        depends = bool((b[0].free_symbols | b[1].free_symbols) & set(shifts))
        return not depends and _precedes(b[1], a[0])

    return all(
        any(
            a is not None and b is not None and apart(a, b)
            for a, b in zip(region, other, strict=False)
        )
        for region in first
        for other in second
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
    pinned_lanes: set[str] = dataclasses.field(default_factory=set)
    # The launch thread axes whose coordinate changes the statement's values
    # (``_propagate_thread_axes``): ``cute.arch.thread_idx()[axis]`` read by
    # the statement or by the definition of a name it reads, the masks
    # excepted.  Along any other axis the statement is the same access on
    # every thread.
    thread_axes: set[int] = dataclasses.field(default_factory=set)
    # For a pinned statement: the locally defined names it reads or writes,
    # any of which its unknown effects may mutate in place (a list passed to
    # a helper).
    touched: frozenset[str] = frozenset()
    # The tensors the statement assigns elements of by subscript.  They are
    # among ``writes`` (the name carries the stored values to later loads)
    # without the statement depending on its own result.
    subscripted: frozenset[str] = frozenset()
    # The tensors mentioned inside a call the pass cannot see through (a
    # helper of unknown purity), which may read or write them however it
    # likes; a pinned statement without such a call (one of a form the pass
    # does not know) has every tensor here.  The other accesses of a pinned
    # statement (the loads and stores of a device loop) are visible.
    opaque: frozenset[str] = frozenset()
    # The tensors mentioned outside those calls: by a plain load or store, a
    # subscript, a pointer built in the open.
    visible: frozenset[str] = frozenset()
    # The regions of the statement's accesses, per tensor, when the memory-op
    # codegen recorded one for every mention of the tensor
    # (``_access_regions``).
    regions: Mapping[str, tuple[_Region, ...]] = dataclasses.field(default_factory=dict)
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
    plain, updated = _subscript_assignments(node)
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
    pinned = not atomic and not _is_movable(node)
    unknown = [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and not (
            _is_plain_store(call)
            or _is_plain_load(call)
            or _is_list_append(call)
            or _is_atomic_call(call)
            or ast.unparse(call.func) in _PURE_CALLS
            or _is_proven_relocatable_call(call, allow_load=True)
        )
    ]
    opaque = frozenset().union(*(_tensor_names(call) for call in unknown))
    visible = _tensor_names_outside(node, unknown)
    if pinned and not unknown:
        opaque = set(_tensor_names(node))
    return _Statement(
        index,
        node,
        canonical(rw.reads),
        canonical(rw.writes),
        tensors,
        canonical(read),
        canonical(written),
        pinned=pinned,
        register_only=not tensors and _is_register_only(node),
        atomic=atomic,
        fence=atomic and not all(_is_relaxed_atomic(call) for call in atomics),
        uniform_lanes=uniform_lanes,
        partly_uniform_lanes=partly_uniform_lanes,
        subscripted=canonical(set(plain) | updated),
        opaque=canonical(opaque),
        visible=canonical(visible),
        regions={
            renames.get(tensor, tensor): regions
            for tensor, regions in _access_regions(node).items()
        },
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
    seeds = [
        {coordinate_lanes[name] for name in s.reads if name in coordinate_lanes}
        for s in everything
    ]
    for statement, lanes in zip(
        everything, _propagate_dependence(everything, seeds, masks), strict=True
    ):
        statement.dataflow_lanes = lanes


def _propagate_dependence(
    everything: list[_Statement], seeds: list[set[_T]], masks: frozenset[str]
) -> list[set[_T]]:
    """Close one dependence set per statement over the names they define and read.

    A statement depends on its seeds and, transitively, on the dependences of
    every statement defining a name it reads.  A tile mask (``masks``)
    carries nothing: it skips a statement in the iterations past the tile's
    end and leaves the address and value of the other iterations alone.
    """
    result = [set(seed) for seed in seeds]
    changed = True
    while changed:
        changed = False
        name_dependences: dict[str, set[_T]] = {}
        for statement, dependences in zip(everything, result, strict=True):
            for name in statement.writes - masks:
                name_dependences.setdefault(name, set()).update(dependences)
        for statement, dependences in zip(everything, result, strict=True):
            size = len(dependences)
            for name in statement.reads:
                dependences |= name_dependences.get(name, set())
            changed = changed or len(dependences) != size
    return result


def _simple_statements(node: ast.AST) -> list[ast.AST]:
    """``node`` itself, or the simple statements of a loop's or branch's bodies.

    The name closures (thread axes, loaded values, a loop's iteration) run
    over these: a device loop already emitted defines its per-iteration
    index from the loop variable and loads its tensor in separate
    statements, and the next loop over the same block reuses the index
    name, so the loop as a whole must not carry the load's taint into it.
    """
    return [statement for statement, _guards in _guarded_statements(node)]


def _guarded_statements(
    node: ast.AST, guards: frozenset[str] = frozenset()
) -> list[tuple[ast.AST, frozenset[str]]]:
    """The simple statements of ``node`` (``_simple_statements``), each with the conjuncts of the branches around it inside ``node``, ``guards`` included.

    A branch's body runs under the conjuncts of its test (``_conjuncts``);
    its else branch and a loop's body under those around the branch or
    loop.
    """
    if isinstance(node, (ast.For, ast.While, ast.If)):
        inside = guards | _conjuncts(node.test) if isinstance(node, ast.If) else guards
        return [
            *(
                pair
                for child in node.body
                for pair in _guarded_statements(child, inside)
            ),
            *(
                pair
                for child in node.orelse
                for pair in _guarded_statements(child, guards)
            ),
        ]
    return [(node, guards)]


def _dependent_names(
    everything: list[_Statement], roots: Iterable[bool], masks: frozenset[str]
) -> set[str]:
    """The names the statements flagged in ``roots`` define, transitively.

    A name defined from one of them (through the names its definition reads)
    is among them too; a tile mask carries nothing (``_propagate_dependence``).
    """
    seeds: list[set[bool]] = [{True} if root else set() for root in roots]
    names: set[str] = set()
    for statement, dependences in zip(
        everything, _propagate_dependence(everything, seeds, masks), strict=True
    ):
        if dependences:
            names |= statement.writes - masks
    return names


def _thread_axis_reads(node: ast.AST) -> set[int]:
    """The launch thread axes ``node`` reads as ``cute.arch.thread_idx()[axis]``."""
    return {
        child.slice.value
        for child in ast.walk(node)
        if isinstance(child, ast.Subscript)
        and isinstance(child.value, ast.Call)
        and ast.unparse(child.value.func) == "cute.arch.thread_idx"
        and isinstance(child.slice, ast.Constant)
        and isinstance(child.slice.value, int)
    }


def _propagate_thread_axes(everything: list[_Statement], masks: frozenset[str]) -> None:
    """Assign every statement the launch thread axes its values depend on.

    A per-lane index reads the thread's coordinate along its tile axis
    (``indices_1 = tile_offset_1 + thread_idx()[0] * 4 + lane_1``, a vec
    wrapper's lane base) and every statement reading it, or a name defined
    from it, addresses an element of that thread's own; a statement reading
    none of them (a ``tile.begin`` access, a scalar gathered by a uniform
    index) is the same access on every thread of the axis.
    """
    seeds = [_thread_axis_reads(statement.node) for statement in everything]
    for statement, axes in zip(
        everything, _propagate_dependence(everything, seeds, masks), strict=True
    ):
        statement.thread_axes = axes


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


def _inexact_repeated_loop(
    loop: _Statement,
    statements: list[_Statement],
    scope: LaneScope,
    attached: list[_Statement],
    renames: Mapping[str, str],
) -> str | None:
    """Why repeating ``loop``, whose values change with ``scope``'s lane, once per lane is observable inside it.

    A loop of the body (a device loop) that reads the lane's values runs
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
    the same order.  Registers are the placement's business: the loop's
    accumulators are initialized within the pass.  A statement of unknown
    effects whose operands ignore the lane may read or write its tensors in
    every pass: it meets any other access of them in the loop.  The
    per-lane accesses are the barrier pass's business, which pairs them
    through their address mappings (``add_thread_barriers``): the check here
    knows the names an access reads, not whether they change its element.
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
    for statement, varies in zip(inside, varying, strict=True):
        if varies:
            continue
        if statement.pinned:
            # A call of unknown effects whose operands ignore the lane runs
            # again in every pass, between every iteration's accesses of its
            # tensors: it may read what a later iteration of the first pass
            # stored, or store over it.
            for other, other_varies in zip(inside, varying, strict=True):
                if other is statement or not (statement.tensors & other.tensors):
                    continue
                tensor = min(statement.tensors & other.tensors)
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
    of the loop's values.  A loop of the body whose values depend on the lane
    loop, or that a barrier or a call of unknown effects inside it pins,
    still repeats the accesses inside it that do not, with every iteration of
    it between two repetitions (``_inexact_repeated_loop``).  Any other
    statement with unknown effects stays where the structure pins it and is
    not checked for repetition; it still orders the checked statements around
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
        for other in scopes[depth + 1 :]:
            if other.lane_var in nested:
                executed.extend(
                    (s, s.pinned or lane_var in s.dataflow_lanes)
                    for s in attached[other.lane_var]
                )
        for statement, varies in executed:
            if varies:
                if statement.index >= 0 and isinstance(statement.node, ast.For):
                    reason = _inexact_repeated_loop(
                        statement, statements, scope, attached[lane_var], renames
                    )
                    if reason is not None:
                        return reason
                continue
            if statement.atomic:
                return _repeated_atomic(statement, lane_var)
            if (statement.reads & statement.writes) - statement.subscripted or (
                statement.tensors_read & statement.tensors_written
            ):
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
    renames: Mapping[str, str],
) -> None:
    reason = _inexact_nest(
        statements, scopes, attached, full_nest=full_nest, renames=renames
    )
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
    _require_exact_nest(
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


_BARRIER_CALL = "cute.arch.sync_threads"


def _is_barrier(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == _BARRIER_CALL
    )


def _register_arrays(nodes: Iterable[ast.AST]) -> set[str]:
    """The names ``nodes`` bind to thread-private register arrays.

    ``cute.make_rmem_tensor`` allocates a thread's own storage (a scan's
    carried prefix): no other thread reads or writes it, so two accesses of
    it never race.
    """
    names: set[str] = set()
    for node in nodes:
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Assign)
                and len(child.targets) == 1
                and isinstance(child.targets[0], ast.Name)
                and isinstance(child.value, ast.Call)
                and ast.unparse(child.value.func) == "cute.make_rmem_tensor"
            ):
                names.add(child.targets[0].id)
    return names


def _races(
    first: _Statement, second: _Statement, axes: set[int], tensors: Iterable[str]
) -> bool:
    """Whether threads differing along ``axes`` may race on a tensor of both statements.

    ``first`` is emitted before ``second``.  A thread's own accesses stay in
    program order, so the pair is safe when ``second`` is a store whose
    address and value ignore ``axes`` and ``first`` does not read the tensor
    (every thread ends with the same store), or when ``first`` is such a
    store and ``second`` reads the tensor or is such a store itself (the
    reader's own store precedes its read; identical stores win in any
    order).  Two atomics commute.  Anything else can be observed in an order
    the program forbids: a read on either side of another thread's write, a
    per-lane write on either side of a uniform one, an atomic (issued by
    the leader thread only) beside a load, a tensor a call of unknown
    effects may touch (``_Statement.opaque``).  ``tensors`` are the tensors
    of both statements to consider (the shared ones; a thread-private
    register array is not among them).
    """
    for tensor in tensors:
        opaque = tensor in first.opaque
        other_opaque = tensor in second.opaque
        writes = opaque or tensor in first.tensors_written
        reads = opaque or tensor in first.tensors_read
        other_writes = other_opaque or tensor in second.tensors_written
        other_reads = other_opaque or tensor in second.tensors_read
        if not (writes or other_writes) or (first.atomic and second.atomic):
            continue
        uniform_store = (
            writes
            and not reads
            and not (first.atomic or opaque)
            and not axes & first.thread_axes
        )
        other_uniform_store = (
            other_writes
            and not other_reads
            and not (second.atomic or other_opaque)
            and not axes & second.thread_axes
        )
        if other_uniform_store and not reads:
            continue
        if uniform_store and (not other_writes or other_uniform_store):
            continue
        return True
    return False


@dataclasses.dataclass
class _Block:
    """One statement list of the emitted nest.

    A list of placement items, the statements of a vectorized loop's wrapper
    (its attached statements and V-loop in emission order, into which
    ``placement.barriers`` positions are applied when the loop is
    materialized), or the body of a branch or while loop among the items.
    The last is ``divergent``: its condition may vary per thread, so no
    barrier can be placed in it (nor in anything nested in it).
    """

    divergent: bool
    items: list[ast.AST | LanePlacement] | None = None
    placement: LanePlacement | None = None
    # The number of children (a wrapper's: its attached statements and V-loop).
    count: int = 0
    # For the body of a branch: the conjuncts of its condition, unparsed.
    conjuncts: frozenset[str] = frozenset()
    # For the statement list one iteration of a lane loop or V-loop runs: the
    # names the loop redefines per iteration, the loop's own variable among
    # them (the lane variable, or the V-loop's) and the conjuncts of the
    # predicate selecting its first iteration (``LaneScope.first_lane``).
    iteration: frozenset[str] = frozenset()
    variables: frozenset[str] = frozenset()
    first: frozenset[str] = frozenset()
    # Positions of the barriers already among the children, and the
    # (after, before) child pairs a barrier has to separate.
    existing: set[int] = dataclasses.field(default_factory=set)
    requests: list[tuple[int, int]] = dataclasses.field(default_factory=list)

    def insert(self, positions: Iterable[int]) -> None:
        for position in sorted(positions, reverse=True):
            if self.placement is not None:
                self.placement.barriers.append(position)
            else:
                assert self.items is not None
                self.items.insert(position, statement_from_string(f"{_BARRIER_CALL}()"))


@dataclasses.dataclass
class _Leaf:
    statement: _Statement
    # The blocks from the root down and the child position in each.
    path: tuple[tuple[_Block, int], ...]
    # For an attached statement of a vectorized loop's wrapper: its scope.
    scope: LaneScope | None = None


def _conjuncts(test: ast.expr) -> frozenset[str]:
    """The conjuncts of ``test`` (itself, or the operands of its ``and``), unparsed."""
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
        return frozenset().union(*(_conjuncts(value) for value in test.values))
    return frozenset({ast.unparse(test)})


def _first_conjuncts(scope: LaneScope, iteration: frozenset[str]) -> frozenset[str]:
    """The conjuncts of the scope's first-iteration predicate reading ``iteration``."""
    if not scope.first_lane:
        return frozenset()
    test = expr_from_string(scope.first_lane)
    assert isinstance(test, ast.expr)
    return frozenset(
        conjunct
        for conjunct in _conjuncts(test)
        if _names_read(expr_from_string(conjunct), {}) & iteration
    )


def _leaves(
    items: list[ast.AST | LanePlacement],
    scopes: Mapping[str, LaneScope],
    renames: Mapping[str, str],
    path: tuple[tuple[_Block, int], ...],
    blocks: list[_Block],
    leaves: list[_Leaf],
    *,
    divergent: bool,
    conjuncts: frozenset[str] = frozenset(),
    iteration: frozenset[str] = frozenset(),
    variables: frozenset[str] = frozenset(),
    first: frozenset[str] = frozenset(),
) -> None:
    """Enumerate the statements of ``items`` in emission order with their paths.

    A branch or while loop among the items is not a statement of its own but
    the divergent block of its body statements (a ``hl.if`` body is emitted
    inside the lane nest without passing through it); a device loop, a
    guard or any other statement is one leaf.  A vectorized loop's wrapper
    block is the body of its lane loop, whose iteration changes every
    coordinate of the scope and whose variable is the lane variable; the
    placement's items are the body of its V-loop, whose iteration changes
    the coordinates its attached statements do not read (the V-loop
    variable, its variable), or of the plain lane loop.
    """
    block = _Block(
        divergent,
        items=items,
        count=len(items),
        conjuncts=conjuncts,
        iteration=iteration,
        variables=variables,
        first=first,
    )
    blocks.append(block)
    for index, item in enumerate(items):
        here = (*path, (block, index))
        if isinstance(item, (ast.If, ast.While)):
            branches = [
                (
                    item.body,
                    _conjuncts(item.test) if isinstance(item, ast.If) else frozenset(),
                ),
                *([(item.orelse, frozenset())] if item.orelse else []),
            ]
            for branch, test in branches:
                _leaves(
                    list(branch),
                    scopes,
                    renames,
                    here,
                    blocks,
                    leaves,
                    divergent=True,
                    conjuncts=test,
                )
            continue
        if not isinstance(item, LanePlacement):
            if _is_barrier(item):
                block.existing.add(index)
            leaves.append(_Leaf(_analyze(-1, item, renames), here))
            continue
        scope = scopes[item.lane_var]
        attached_reads: set[str] = set()
        for node in scope.attached:
            attached_reads |= _names_read(node, renames)
        inner = frozenset(scope.coordinates - attached_reads) or scope.coordinates
        outer = scope.coordinates if scope.attached else frozenset()
        wrapper = _Block(
            divergent,
            placement=item,
            count=len(scope.attached) + 1,
            iteration=outer,
            variables=frozenset({item.lane_var}) if outer else frozenset(),
            first=_first_conjuncts(scope, outer),
        )
        blocks.append(wrapper)
        for position, node in enumerate(scope.attached[: scope.vloop_index]):
            leaves.append(
                _Leaf(_analyze(-1, node, renames), (*here, (wrapper, position)), scope)
            )
        _leaves(
            item.items,
            scopes,
            renames,
            (*here, (wrapper, scope.vloop_index)),
            blocks,
            leaves,
            divergent=divergent,
            iteration=inner,
            variables=frozenset(name for name in inner if name in scope.counts),
            first=_first_conjuncts(scope, inner),
        )
        for position, node in enumerate(
            scope.attached[scope.vloop_index :], start=scope.vloop_index + 1
        ):
            leaves.append(
                _Leaf(_analyze(-1, node, renames), (*here, (wrapper, position)), scope)
            )


def _common_block(first: _Leaf, second: _Leaf) -> tuple[_Block, int, int]:
    """The nearest statement list holding both leaves and their child positions in it.

    Leaves in the two branches of one ``if`` share their path down to the
    branch statement and part in blocks that are not one list: they are
    reported at the branch statement with equal positions, which no barrier
    can separate (threads taking different branches race unordered).
    """
    parent: tuple[_Block, int] | None = None
    for (block, index), (other, other_index) in zip(
        first.path, second.path, strict=False
    ):
        if block is not other:
            assert parent is not None
            return parent[0], parent[1], parent[1]
        if index != other_index:
            return block, index, other_index
        parent = (block, index)
    raise AssertionError("two leaves share a path")


def _below(
    path: tuple[tuple[_Block, int], ...], level: _Block
) -> tuple[tuple[_Block, int], ...]:
    """The part of ``path`` from ``level`` down."""
    return path[
        next(index for index, (block, _) in enumerate(path) if block is level) :
    ]


def add_thread_barriers(
    items: list[ast.AST | LanePlacement],
    scopes: Sequence[LaneScope],
    *,
    rename_groups: Mapping[str, str],
    axis_sizes: Mapping[int, int],
    definitions: Iterable[ast.AST],
    masks: frozenset[str],
    allow_barriers: bool = True,
    loop_variables: frozenset[str] = frozenset(),
    loop_body: list[ast.AST | LanePlacement] | None = None,
    loop_shifts: _Shifts | None = None,
    constants: Mapping[str, int] | None = None,
    loop_headers: Sequence[ast.For] = (),
) -> None:
    """Separate with block-wide barriers the tensor accesses threads could race on.

    ``items`` is the nest about to be emitted (placement items and
    ``LanePlacement`` loops, or the body statements themselves when no lane
    loop is live); ``scopes`` describes the live loops with their attached
    statements and trip counts; ``axis_sizes`` gives the launch thread count
    along each thread axis the body may address; ``definitions`` are the
    per-thread index and mask definitions around the body (the lane setup,
    the thread axes without a lane loop, the vec wrappers' lane bases) and
    ``masks`` the tile masks among them; ``constants`` the values of the
    block-size constants the loop bounds inside the body name;
    ``loop_headers`` the ``for`` statements of the device loop whose body
    the nest is (their bounds model the loop's variables across
    iterations) and of the loops around it (the enclosing device loops and
    lane loops: their variables are one value each throughout the body,
    nonnegative and below their bounds, ``AddressMaps``).  Two
    statements accessing one tensor, at least one writing it, whose threads
    may differ along an axis shared by several threads (the address
    mappings of the two accesses do not force the coordinate equal,
    ``AddressMaps.same_thread``, the indices the tile masks around each
    access bound counting as bounded, ``AddressMaps.bounds``: a uniform
    address, a gathered one, a
    neighbour's or a permuted element) get a barrier between them at their
    nearest common statement list unless ``_races`` proves the order
    immaterial or the recorded regions of their accesses are apart
    (``_disjoint``); a barrier
    already there counts.  Modifies ``items`` in place: barrier statements
    are inserted among placement items, and positions among a vectorized
    loop's wrapper statements are recorded on its ``LanePlacement.barriers``.
    A pair whose barrier would sit in control flow a thread may skip, where
    a block-wide barrier deadlocks (the body of a branch or while loop among
    the items, or the whole nest without ``allow_barriers``), is rejected
    with ``BackendUnsupported`` instead.

    Every loop of the nest runs its body once per iteration on every thread
    with nothing between the iterations.  The device loop whose body the
    nest is (``loop_variables`` names what it redefines per iteration, its
    ``for`` targets, and ``loop_body`` the statement list one iteration runs:
    ``items``, or the innermost loop's) runs its iterations one after the
    other in the program: two statements of its body, in either emission
    order and a statement with itself included on a tensor it both reads
    and writes (a device loop nested in the body holds a uniform read and a
    per-thread store of its own; its unknown effects, the tensors inside
    calls the pass cannot see through, are not ordered against its own next
    instance) or only stores to through a map the terms know whole (which
    folds between the iterations when the thread and the iteration sum into
    one index), racing as one iteration's and a later one's, need a barrier
    after the first
    within its iteration or before the second within its own; when none of
    the placed ones does, one is added at the end of the body (or right
    after the first statement, when that sits among a vectorized loop's
    wrapper statements past the body).  Only the recorded regions of the
    accesses exempt such a pair, when they cannot meet in any later
    iteration (``loop_shifts`` maps the loop's begin symbols to their step,
    ``_disjoint``), or the address mappings, when they force the thread
    equal with the loop's variables taken as differing between the two
    iterations; a uniform access whose address moves with the iteration
    may still meet the previous iteration's per-thread stores.  A lane
    loop's or V-loop's iterations are one tile statement's lanes, run
    together before the next statement's, on one thread whatever the thread
    count along its axis: such a pair, when one side writes and a gathered
    address is involved or both addresses vary with the lane without their
    equal terms forcing the loop's own variable equal, is rejected with
    ``BackendUnsupported`` instead; so is a device loop among the items,
    which each lane runs whole, whose body accesses one tensor twice (one
    statement's own load and store included, and its stores alone through
    a map the terms know whole) with a write, unless the pair is confined
    to a lane by its terms, never meets (regions count by their literal
    bounds only there), is two stores whose addresses ignore the lane, or
    is one statement's stores confined to an iteration.  A packet (a vector
    load, a store flush) is
    every one of its elements, one per iteration of its wrapper's V-loop
    (``_packets``, or the scope of the wrapper it is attached to); a packet
    outside any wrapper the pass recognizes is an access of unknown
    elements, which forces nothing.  A statement guarded by a loop's
    first-iteration predicate (a pinned atomic) has no instance in the next
    iteration and, separated by a barrier from a later one, is ordered
    before all of its lanes.
    """
    shared = {axis for axis, size in axis_sizes.items() if size > 1}
    definitions = list(definitions)
    if not items:
        return
    by_lane = {scope.lane_var: scope for scope in scopes}
    blocks: list[_Block] = []
    leaves: list[_Leaf] = []
    _leaves(
        items, by_lane, rename_groups, (), blocks, leaves, divergent=not allow_barriers
    )
    if contains_compiler_marker([leaf.statement.node for leaf in leaves]):
        return
    statements = [leaf.statement for leaf in leaves]
    # The closures run over simple statements: a compound leaf (a device
    # loop) is represented by the statements of its body and takes their
    # thread axes together.
    seen: set[int] = set()
    everything: list[_Statement] = []
    parts: list[tuple[_Statement, list[_Statement]]] = []
    for statement in statements:
        pieces = _simple_statements(statement.node)
        if pieces == [statement.node]:
            analyzed = [statement]
        else:
            analyzed = [_analyze(-1, piece, rename_groups) for piece in pieces]
            parts.append((statement, analyzed))
        for piece_statement in analyzed:
            if id(piece_statement.node) not in seen:
                seen.add(id(piece_statement.node))
                everything.append(piece_statement)
    for node in definitions:
        for piece in _simple_statements(node):
            if id(piece) not in seen:
                seen.add(id(piece))
                everything.append(_analyze(-1, piece, rename_groups))
    _propagate_thread_axes(everything, masks)
    for statement, analyzed in parts:
        statement.thread_axes = set().union(*(piece.thread_axes for piece in analyzed))
    private = {
        rename_groups.get(name, name)
        for name in _register_arrays(statement.node for statement in everything)
    }
    # A value loaded from memory (by a load, an atomic, or a call handed a
    # tensor) may be any thread's index: an address term reading it, or a
    # name defined from it, is not the thread's own element (nor confined to
    # a loop iteration).  A shape query or a thread coordinate is no load.
    # The address maps translate a name defined from a loaded value through
    # its definition, down to the loaded names themselves (unknown integers
    # of their own): the part of a term computed from the thread's
    # coordinate is still the thread's.
    roots = [
        bool(statement.tensors_read) or statement.atomic for statement in everything
    ]
    loaded = _dependent_names(everything, roots, masks)
    loaded_names = {
        name
        for statement, root in zip(everything, roots, strict=True)
        if root
        for name in statement.writes - masks
    }
    # The loads among the definitions by the name each defines, and the
    # tensors anything in view writes or may (a call of unknown effects, an
    # atomic): a load of any other kernel argument at a uniform address is
    # one value on every thread and lane (``AddressMaps.uniform_load``).
    loads: dict[str, Load] = {}
    for node in definitions:
        for piece in _simple_statements(node):
            defining = _defining_load(piece, rename_groups)
            if defining is not None:
                loads[defining[0]] = defining[1]
    written_tensors = frozenset().union(
        *(
            statement.tensors_written
            | statement.opaque
            | (statement.tensors if statement.atomic else frozenset())
            for statement in everything
        )
    )
    # Per statement and tensor, the names read by each term of each address,
    # and the terms themselves over the thread coordinates, the loop
    # variables and the uniform values (``AddressMaps``).
    maps = AddressMaps(
        axis_sizes=axis_sizes,
        loop_counts={
            name: count for scope in scopes for name, count in scope.counts.items()
        },
        leaves=[leaf.statement.node for leaf in leaves],
        definitions=definitions,
        renames=rename_groups,
        loaded=loaded_names,
        constants=constants or {},
        outer_loops=loop_headers,
        loads=loads,
        written=written_tensors,
    )

    def canonical(name: str) -> str:
        return rename_groups.get(name, name)

    def modelled(
        node: ast.AST,
        owner: ast.AST,
        packets: Mapping[int, tuple[str, str]],
        scope: LaneScope | None,
        bounded: Mapping[str, frozenset[sympy.Expr]] | frozenset[sympy.Expr],
    ) -> dict[str, list[list[sympy.Expr | None]] | None]:
        """Per tensor of ``node``, the terms of each access over the coordinates; None for a tensor with an access of unknown elements.

        ``owner`` is the leaf whose definitions the terms read (``node`` or
        the compound statement holding it), ``packets`` the wrappers'
        packets in view, ``scope`` the vectorized loop whose wrapper
        ``node`` is a statement of, if any: a packet of theirs covers one
        element per iteration of the V-loop along the dimension the
        per-thread base indexes (``AddressMaps.packet``); ``bounded`` the
        indices the branches around ``node`` bound, kept whole in the terms
        (``AddressMaps.bounds``), for every tensor or per tensor (the
        branches inside a compound statement around every access of a
        tensor, ``inner_guards``).
        """

        def terms_of(tensor: str, access: _Access) -> list[sympy.Expr | None] | None:
            terms = flat_terms(_address_terms(access.address, tensor))
            quantities = (
                bounded
                if isinstance(bounded, frozenset)
                else bounded.get(canonical(tensor), frozenset())
            )
            values = [maps.translate(term, owner, quantities) for term in terms]
            if access.element:
                return values
            if access.vector is not None and access.base is not None:
                packet: tuple[str, str] | None = (access.vector, access.base)
            else:
                packet = _scope_packet(scope) if scope is not None else None
            if packet is None:
                return None
            vector, base = (canonical(name) for name in packet)
            along = [
                index
                for index, term in enumerate(terms)
                if base in _names_read(term, rename_groups)
            ]
            if len(along) != 1:
                return None
            (index,) = along
            values[index] = maps.packet(values[index], vector)
            return None if values[index] is None else values

        result: dict[str, list[list[sympy.Expr | None]] | None] = {}
        for tensor, accesses in _access_addresses(node, packets).items():
            values = [terms_of(tensor, access) for access in accesses]
            result[canonical(tensor)] = (
                None
                if any(value is None for value in values)
                else [value for value in values if value is not None]
            )
        return result

    address_terms: dict[int, dict[str, list[list[set[str]]]]] = {}
    # Per statement and tensor: each access's terms over the coordinates,
    # every element of it modelled (``modelled``).
    address_maps: dict[int, dict[str, list[list[sympy.Expr | None]] | None]] = {}
    # The packets of the wrappers inside each compound leaf.
    packets_inside = {
        id(leaf.statement): _packets(leaf.statement.node) for leaf in leaves
    }

    def outside_accesses(statement: _Statement) -> set[str]:
        """The tensors ``statement`` accesses outside its simple statements.

        An access in a branch's test, a while's test or a loop's header
        (``_guarded_statements`` yields the bodies' statements alone) counts
        for the statement as a whole but belongs to none of its parts, so
        the pairs over the parts never see it.
        """
        packets = packets_inside[id(statement)]
        counted: collections.Counter[str] = collections.Counter()
        for inner, _guards in _guarded_statements(statement.node):
            for tensor, accesses in _access_addresses(inner, packets).items():
                counted[canonical(tensor)] += len(accesses)
        total: collections.Counter[str] = collections.Counter()
        for tensor, accesses in _access_addresses(statement.node, packets).items():
            total[canonical(tensor)] += len(accesses)
        return {name for name, count in total.items() if count != counted[name]}

    def inner_guards(statement: _Statement) -> dict[str, frozenset[str]]:
        """Per tensor ``statement`` accesses, the conjuncts of the branches inside it around every access of the tensor.

        A compound statement's accesses run under its own branches (the
        column tile's mask around a device loop's store): a conjunct
        around each access of a tensor bounds the tensor's indices for
        the statement as a whole (``AddressMaps.bounds``), as it does for
        each access.  A tensor with an access outside the statement's
        simple statements (in a branch's test, a loop's header) shares
        nothing; nor does any tensor of a simple statement.
        """
        packets = packets_inside[id(statement)]
        common: dict[str, frozenset[str]] = {}
        counted: collections.Counter[str] = collections.Counter()
        for inner, guards in _guarded_statements(statement.node):
            for tensor, accesses in _access_addresses(inner, packets).items():
                name = canonical(tensor)
                counted[name] += len(accesses)
                common[name] = common[name] & guards if name in common else guards
        total: collections.Counter[str] = collections.Counter()
        for tensor, accesses in _access_addresses(statement.node, packets).items():
            total[canonical(tensor)] += len(accesses)
        return {
            name: common[name] if counted[name] == count else frozenset()
            for name, count in total.items()
        }

    # The thread coordinates and the indices the branches around each
    # statement bound, per tensor it accesses (``AddressMaps.bounds``): the
    # branches around the statement, and those inside a compound statement
    # around every access of the tensor (``inner_guards``).  An index
    # bounded by a literal is one quantity of the statement's terms
    # (``whole_quantities``).
    bounds: dict[int, dict[str, dict[sympy.Expr, Bound]]] = {}
    for leaf in leaves:
        around = [
            conjunct for block, _index in leaf.path for conjunct in block.conjuncts
        ]
        bounds[id(leaf.statement)] = {
            name: maps.bounds(leaf.statement.node, around, guards)
            for name, guards in inner_guards(leaf.statement).items()
        }
    for leaf in leaves:
        statement = leaf.statement
        per_tensor = address_terms[id(statement)] = {}
        for tensor, accesses in _access_addresses(statement.node).items():
            name = canonical(tensor)
            per_tensor[name] = [
                [
                    _names_read(term, rename_groups) - {name}
                    for term in _address_terms(access.address, tensor)
                ]
                for access in accesses
            ]
        address_maps[id(statement)] = modelled(
            statement.node,
            statement.node,
            packets_inside[id(statement)],
            leaf.scope,
            {
                name: whole_quantities(quantities)
                for name, quantities in bounds[id(statement)].items()
            },
        )

    def forced_equal(
        mine: list[list[sympy.Expr | None]] | None,
        theirs: list[list[sympy.Expr | None]] | None,
        ranges: Mapping[sympy.Expr, Bound | None],
        varying: Iterable[str] = (),
    ) -> frozenset[sympy.Symbol]:
        """The variables every pair of accesses, one of ``mine`` and one of ``theirs``, forces equal (``AddressMaps.same_thread``)."""
        if mine is None or theirs is None:
            return frozenset()
        result: frozenset[sympy.Symbol] | None = None
        for access in mine:
            for other in theirs:
                if len(access) != len(other):
                    return frozenset()
                equal = maps.same_thread(access, other, ranges=ranges, varying=varying)
                result = equal if result is None else result & equal
                if not result:
                    return frozenset()
        return result or frozenset()

    def same_thread(
        first: _Statement, second: _Statement, tensor: str, varying: Iterable[str]
    ) -> frozenset[sympy.Symbol]:
        """The variables every pair of accesses of ``tensor``, one from each statement, forces equal.

        A coordinate among them: along its axis the two statements meet on
        one thread only.  A lane variable: the two statements meet on one
        lane only.  Empty when an access is unknown (a tensor inside a call
        the pass cannot see through, a packet outside any wrapper the pass
        recognizes) or any pair of accesses may meet on different threads
        and lanes.  ``varying`` names the uniform values that differ between
        the two statements' instances (the device loop's variables, between
        two iterations).
        """
        if tensor in first.opaque or tensor in second.opaque:
            return frozenset()
        mine = address_maps.get(id(first), {}).get(tensor)
        theirs = address_maps.get(id(second), {}).get(tensor)
        if mine is None or theirs is None:
            return frozenset()
        return forced_equal(
            mine,
            theirs,
            maps.pair_ranges(bounds[id(first)][tensor], bounds[id(second)][tensor]),
            varying,
        )

    # The iterations of the device loop's variables, when their headers model
    # them: two accesses whose equal terms force them equal meet within one
    # iteration only.
    iterations = {
        iteration_symbol(name)
        for name in loop_variables
        if uniform_symbol(name) in maps.iteration_values
    }

    def apart(
        first: _Statement, second: _Statement, tensor: str, shifts: _Shifts | None
    ) -> bool:
        """Whether the recorded regions prove the two accesses of ``tensor`` never meet."""
        regions = first.regions.get(tensor)
        other = second.regions.get(tensor)
        return (
            regions is not None
            and other is not None
            and _disjoint(regions, other, shifts=shifts)
        )

    def racing(
        first: _Statement,
        second: _Statement,
        tensor: str,
        *,
        shifts: _Shifts | None,
        varying: Iterable[str] = (),
    ) -> bool:
        """Whether ``first``, run before ``second`` (in a later iteration under ``shifts``), may race with it on ``tensor``.

        ``varying`` names the values differing between the two statements'
        instances (the device loop's variables, across iterations).
        """
        if not _races(first, second, shared, [tensor]):
            # No write, or an order the program allows whatever the threads.
            return False
        if apart(first, second, tensor, shifts):
            return False
        equal = same_thread(first, second, tensor, varying)
        if varying and iterations and iterations <= equal:
            # The accesses meet within one iteration only, whose pairs the
            # thread level orders.
            return False
        differing = {axis for axis in shared if thread_symbol(axis) not in equal}
        return bool(differing) and _races(first, second, differing, [tensor])

    def conflicting(
        first: _Statement, second: _Statement, tensor: str, shifts: _Shifts | None
    ) -> bool:
        """Whether lanes of one thread, ``first``'s then ``second``'s, may touch one element of ``tensor`` with a write.

        Threads play no part: the lanes run on one thread in loop order,
        while the program runs every lane of ``first`` before any of
        ``second``, so a write on either side (two atomics excepted, which
        commute) can be observed out of program order.  ``shifts`` is for
        the recorded regions (``_disjoint``): two statements of the lane
        loop's body share every tile begin (``{}``); the statements of a
        device loop the lanes repeat whole share the literal bounds only
        (None), as the loop's own tile begins differ between the iterations
        the two passes pair and the pass does not tell them from the
        enclosing tiles'.
        """
        if apart(first, second, tensor, shifts):
            return False
        writes = tensor in first.opaque or tensor in first.tensors_written
        other_writes = tensor in second.opaque or tensor in second.tensors_written
        return (writes or other_writes) and not (first.atomic and second.atomic)

    def reject(tensors: Iterable[str]) -> None:
        raise exc.BackendUnsupported(
            "cute",
            f"threads sharing a tile axis race on {', '.join(sorted(tensors))} "
            "inside a branch, where the barrier ordering them cannot be placed",
        )

    for position, first in enumerate(leaves):
        if not first.statement.tensors - private:
            continue
        for second in leaves[position + 1 :]:
            shared_tensors = (
                first.statement.tensors & second.statement.tensors
            ) - private
            tensors = {
                tensor
                for tensor in shared_tensors
                if racing(first.statement, second.statement, tensor, shifts={})
            }
            if tensors:
                block, after, before = _common_block(first, second)
                if any(after < barrier <= before for barrier in block.existing):
                    # Ordered already, by a barrier the lowering emitted
                    # itself (a scan's, between its shared-memory phases).
                    continue
                if block.divergent or after == before:
                    reject(tensors)
                block.requests.append((after, before))

    # The loops of the nest: (the block whose subtree one iteration runs,
    # the names changing per iteration, the first-iteration conjuncts, the
    # block whose end closes an iteration, whether only literal region bounds
    # count).  The device loop's tail is the innermost body: the lane nest
    # around it runs whole inside each iteration, and its top-level list may
    # be a stand-in for the body.
    # The lane loops and V-loops of the nest, innermost first (a barrier
    # closing an inner loop's body closes the enclosing loops' iterations
    # too), then the device loop.  The device loop's iterations follow each
    # other in the program: a barrier closing its body orders them.  A lane
    # loop emulates one tile-level statement per iteration, whose lanes the
    # program runs together before the next statement's: two lanes of it
    # cannot be ordered by a barrier, so a pair conflicting across lanes is
    # rejected -- on one thread, whatever the thread count along the axis.
    # Two accesses whose equal terms force the lane equal are confined to
    # their lane and never conflict, and an access invariant along the loop
    # is the exactness check's business (``_inexact_nest``); the pairs left
    # are those a gathered address makes conflict, and those of two
    # addresses varying with the lane through different mappings (the
    # shifted read beside the plain store).  The device loop's tail is
    # the innermost body: the lane nest around it runs whole inside each
    # iteration, and its top-level list may be a stand-in for the body.
    levels: list[tuple[_Block, frozenset[str], frozenset[str], _Block, bool]] = [
        (block, block.iteration, block.variables, block, False)
        for block in reversed(blocks)
        if block.iteration
    ]
    if loop_variables:
        body = (
            blocks[0]
            if loop_body is None
            else next(block for block in blocks if block.items is loop_body)
        )
        levels.append((blocks[0], loop_variables, frozenset(), body, True))

    def gathered(statement: _Statement, tensor: str) -> bool:
        """Whether a term of an address of ``tensor`` reads a value loaded from memory."""
        accesses = address_terms.get(id(statement), {}).get(tensor)
        return accesses is not None and any(
            names & loaded for terms in accesses for names in terms
        )

    def varies(statement: _Statement, tensor: str, varying: set[str]) -> bool:
        """Whether every access of ``tensor`` is confined to its lane by a term."""
        accesses = address_terms.get(id(statement), {}).get(tensor)
        if accesses is None or tensor in statement.opaque:
            return False
        return all(
            any(not (names & loaded) and names & varying for names in terms)
            for terms in accesses
        )

    def first_only(leaf: _Leaf, level: _Block) -> bool:
        return bool(level.first) and any(
            block.conjuncts & level.first for block, _index in _below(leaf.path, level)
        )

    inner_parts = {id(statement): analyzed for statement, analyzed in parts}

    def invariant(
        accesses: list[list[sympy.Expr | None]] | None, symbols: set[sympy.Symbol]
    ) -> bool:
        """Whether every access is known and reads none of ``symbols``, nor any unknown value, in any term."""
        return accesses is not None and all(
            term is not None and not term.free_symbols & (symbols | maps.unknowns)
            for terms in accesses
            for term in terms
        )

    def stores_only(statement: _Statement, tensor: str) -> bool:
        return (
            tensor in statement.tensors_written
            and tensor not in statement.tensors_read
            and tensor not in statement.opaque
            and not statement.atomic
        )

    def own_pairs(
        statement: _Statement,
        model: Mapping[str, list[list[sympy.Expr | None]] | None],
    ) -> frozenset[str]:
        """The tensors ``statement`` is paired with itself on: those it both reads and writes, and those it only stores to through addresses the terms model whole.

        A read on one side of the statement's own write is what the order
        of two instances decides; two of its stores land in program order
        within one instance and meet across instances (two lanes' passes,
        two threads' iterations) only when the map from the instance to the
        element folds, which the terms show when they know every value in
        them.  A store through a value the pass does not know (a jagged
        tile's rows begin at an offset loaded per row) is not modelled.
        """
        return frozenset(
            tensor
            for tensor in statement.tensors & statement.visible
            if (
                tensor in statement.tensors_read | statement.opaque
                and tensor in statement.tensors_written | statement.opaque
            )
            or (stores_only(statement, tensor) and invariant(model.get(tensor), set()))
        )

    def repeated_whole(
        leaf: _Leaf, variables: frozenset[str], level_symbols: set[sympy.Symbol]
    ) -> None:
        """Reject a compound leaf whose body's accesses two lanes of the loop of ``variables`` interleave.

        A device loop under a lane loop runs whole once per lane: the second
        lane's pass runs every iteration of it after the first lane's did.
        Two accesses of one tensor inside it, one writing, are one lane's
        business when their equal terms force the lane equal, or never meet
        (terms that are never equal; regions apart by their literal bounds,
        the loop's own tile begins differing between the iterations the
        passes pair), and two stores whose addresses ignore the lane
        re-apply the same values in program order; any other pair (another
        lane's element read or written through a different mapping, an
        address every lane shares beside a per-lane one, a call the pass
        cannot see through) is observed in an order the program forbids,
        which no barrier restores.  A statement meets itself on a tensor it
        both loads and stores (a read-modify-write through the lanes'
        mappings), and on one it only stores to when the lane-to-element
        map folds between the loop's iterations (the lane and the iteration
        summed into one index: the second pass's earlier iteration lands
        on the first pass's later one, which the program ordered the other
        way); its stores confined to a lane, or to one iteration of the
        loop (a collision within one tile statement is the program's own),
        are not (``own_pairs``).  The tile masks guarding a statement bound
        the indices they name (``AddressMaps.bounds``): the flattened
        ``n * row + column`` under ``column < n`` is one row's whatever
        the column tile's padding, with ``n`` a literal or a kernel
        argument (a dynamic shape's size).  A tensor the loop reads in a
        branch's test, a while's test or a loop's header beside a store to
        it is accessed where the pairs over the body's statements do not
        reach (``outside_accesses``), and the loop is rejected.
        """
        inner = inner_parts.get(id(leaf.statement))
        if inner is None or not isinstance(leaf.statement.node, ast.For):
            # A guarded statement (a masked flush) runs once per lane like
            # a simple one; only a loop repeats its body's accesses whole.
            return
        packets = packets_inside[id(leaf.statement)]
        exposed = sorted(
            (outside_accesses(leaf.statement) & leaf.statement.tensors_written)
            - private
        )
        if exposed:
            raise exc.BackendUnsupported(
                "cute",
                f"a loop repeated whole once per {', '.join(sorted(variables))} "
                f"accesses {exposed[0]} in a branch test or a loop header beside "
                "a store to it, which the lanes' passes interleave in an order "
                "the pass cannot pair (the loop would have to be split)",
            )
        # Each statement of the body runs under the branches around the
        # leaf (read before the leaf runs, over the definitions reaching
        # it) and under those inside it around the statement (over the
        # leaf's own definitions).
        around = [
            conjunct for block, _index in leaf.path for conjunct in block.conjuncts
        ]
        inner_bounds = [
            maps.bounds(leaf.statement.node, around, guards)
            for _statement, guards in _guarded_statements(leaf.statement.node)
        ]
        models = [
            modelled(
                statement.node,
                leaf.statement.node,
                packets,
                leaf.scope,
                whole_quantities(bounded),
            )
            for statement, bounded in zip(inner, inner_bounds, strict=True)
        ]
        # The iterations of the device loops the leaf runs, as the proof
        # knows them (their variables, bound by their headers).
        iteration_symbols = {
            loop_symbol(canonical(loop.target.id))
            for loop in ast.walk(leaf.statement.node)
            if isinstance(loop, ast.For)
            and _is_device_loop(loop)
            and isinstance(loop.target, ast.Name)
        } & set(maps.ranges)
        for position, (first, mine) in enumerate(zip(inner, models, strict=True)):
            for offset, (second, theirs) in enumerate(
                zip(inner[position:], models[position:], strict=True)
            ):
                shared_tensors = (first.tensors & second.tensors) - private
                if second is first:
                    shared_tensors &= own_pairs(first, mine)
                ranges = maps.pair_ranges(
                    inner_bounds[position], inner_bounds[position + offset]
                )
                for tensor in sorted(shared_tensors):
                    if not conflicting(first, second, tensor, None):
                        continue
                    if tensor in first.opaque or tensor in second.opaque:
                        equal: frozenset[sympy.Symbol] = frozenset()
                    else:
                        equal = forced_equal(
                            mine.get(tensor), theirs.get(tensor), ranges
                        )
                    if level_symbols and level_symbols <= equal:
                        continue
                    if (
                        second is first
                        and stores_only(first, tensor)
                        and iteration_symbols
                        and iteration_symbols <= equal
                    ):
                        continue
                    if (
                        invariant(mine.get(tensor), level_symbols)
                        and invariant(theirs.get(tensor), level_symbols)
                        and stores_only(first, tensor)
                        and stores_only(second, tensor)
                    ):
                        continue
                    raise exc.BackendUnsupported(
                        "cute",
                        f"a loop repeated whole once per {', '.join(sorted(variables))} "
                        f"accesses {tensor} in an order the lanes' passes interleave: "
                        f"{ast.unparse(first.node).splitlines()[0][:80]!r} and "
                        f"{ast.unparse(second.node).splitlines()[0][:80]!r} are not "
                        "confined to one lane, which no barrier orders as the program "
                        "does (the loop would have to be split)",
                    )

    # (first in iteration j, second in a later iteration, the loop, its tail,
    # whether the iterations follow each other, the tensors).  A statement
    # meets itself across the device loop's iterations too, on a tensor it
    # both reads and writes (a nested loop's uniform read and per-thread
    # store): a read on one side of another thread's write is what a barrier
    # orders, while two stores of one statement land in program order on
    # each thread and meet across threads only when the thread-to-element
    # map moves between iterations (a persistent loop's tile-granular
    # offsets do not; a row index summed with the iteration does), which
    # the terms show when they know every value in them (``own_pairs``).
    # A tensor the
    # statement touches only inside calls the pass cannot see through is
    # that call's own business against its next instance (a TMA copy or an
    # MMA issue loop of an emitter runs every iteration and synchronizes its
    # own pipeline); beside a visible load or store of the tensor such a
    # call counts as a read and a write of it, like any unknown access.
    # Across the lanes of one lane loop a statement's own accesses are one
    # tile statement's.
    wraps: list[tuple[_Leaf, _Leaf, _Block, _Block, bool, set[str]]] = []
    for level, seeds, variables, tail, sequential in levels:
        varying = set(seeds) | _dependent_names(
            everything,
            (bool(statement.reads & seeds) for statement in everything),
            masks,
        )
        # The loop's own variable, as the proof knows it: a pair is confined
        # to an iteration of the loop when its equal terms force it equal
        # (the V-loop's variable forced equal confines a pair to a vector
        # lane, not to a lane of the loop around it).
        level_symbols = {
            loop_symbol(name) for name in variables if loop_symbol(name) in maps.ranges
        }
        under = [
            leaf for leaf in leaves if any(block is level for block, _ in leaf.path)
        ]
        for first in under:
            if not first.statement.tensors - private:
                continue
            for second in under:
                if first_only(second, level):
                    continue
                if second is first and not sequential:
                    repeated_whole(first, variables, level_symbols)
                    continue
                shared_tensors = (
                    first.statement.tensors & second.statement.tensors
                ) - private
                if second is first:
                    shared_tensors &= own_pairs(
                        first.statement, address_maps[id(first.statement)]
                    )
                if sequential:
                    tensors = {
                        tensor
                        for tensor in shared_tensors
                        if racing(
                            first.statement,
                            second.statement,
                            tensor,
                            shifts=loop_shifts,
                            varying=varying,
                        )
                    }
                else:
                    tensors = {
                        tensor
                        for tensor in shared_tensors
                        if conflicting(first.statement, second.statement, tensor, {})
                        and (
                            gathered(first.statement, tensor)
                            or gathered(second.statement, tensor)
                            or (
                                varies(first.statement, tensor, varying)
                                and varies(second.statement, tensor, varying)
                            )
                        )
                        and not (
                            level_symbols
                            and level_symbols
                            <= same_thread(
                                first.statement, second.statement, tensor, ()
                            )
                        )
                    }
                if tensors:
                    wraps.append((first, second, level, tail, sequential, tensors))

    placed: dict[int, list[int]] = {id(block): [] for block in blocks}
    for block in blocks:
        for after, before in sorted(block.requests, key=operator.itemgetter(1)):
            barriers = [*block.existing, *placed[id(block)]]
            if not any(after < barrier <= before for barrier in barriers):
                placed[id(block)].append(before)

    def barriers_of(block: _Block) -> list[int]:
        return [*block.existing, *placed[id(block)]]

    def separated(first: _Leaf, second: _Leaf, level: _Block) -> bool:
        # A barrier after ``first``'s subtree or before ``second``'s in a
        # statement list of the loop every thread runs, within one iteration.
        for block, index in _below(first.path, level):
            if block.divergent:
                break
            if any(barrier > index for barrier in barriers_of(block)):
                return True
        for block, index in _below(second.path, level):
            if block.divergent:
                break
            if any(barrier <= index for barrier in barriers_of(block)):
                return True
        return False

    for first, second, level, tail, sequential, tensors in wraps:
        if separated(first, second, level) and (sequential or first_only(first, level)):
            continue
        if not sequential:
            raise exc.BackendUnsupported(
                "cute",
                f"an access of {', '.join(sorted(tensors))} (gathered, or another "
                "lane's element) conflicts with another access of it across the "
                "lanes of one lane loop, which no barrier orders as the program "
                "does (the loop would have to be split)",
            )
        if tail.divergent:
            reject(tensors)
        if any(block is tail for block, _index in first.path):
            block, position = tail, tail.count
        else:
            block, index = next(
                (block, index)
                for block, index in reversed(_below(first.path, level))
                if not block.divergent
            )
            position = index + 1
        placed[id(block)].append(position)
    for block in blocks:
        for position in sorted(placed[id(block)]):
            log.debug("thread barrier placed %s", _describe(block, position))
        block.insert(placed[id(block)])


def _describe(block: _Block, position: int) -> str:
    """Where a barrier at child ``position`` of ``block`` goes, for the log."""
    if block.placement is not None:
        return f"in the wrapper of {block.placement.lane_var} before child {position}"
    assert block.items is not None
    if position >= len(block.items):
        return "closing the statement list"
    item = block.items[position]
    if isinstance(item, LanePlacement):
        return f"before the lane loop of {item.lane_var}"
    return f"before {ast.unparse(item).splitlines()[0][:80]!r}"


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
