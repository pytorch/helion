"""Thread barriers between the accesses of one tensor in a lane-loop nest.

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
loop around it).  A looped reduction's lane loop and V-loop around a device
loop's body, which the nest does not hold, run such lanes too, one chunk at
a time: a pair that may meet in two lanes of one chunk is rejected the same
way (``lane_variables``).  A device loop inside a lane loop runs whole once
per lane, so its body's accesses of one tensor are paired the same way, with
the device loop's iteration free on both sides: the previous row's element
read per lane beside the row's store (``x[t0.index - 1, tk]`` and
``x[t0, tk]``), or an address every lane shares beside a per-lane one, is
rejected; two accesses confined to a lane, two that never meet (only the
literal bounds of their recorded regions count there: the loop's own tile
begins differ between the iterations the passes pair), or two stores whose
addresses ignore the lane (re-applied in program order) are not.  A
statement's own stores meet each other, across the lanes' passes as across
the device loop's iterations, when the map from the instance to the element
folds (the lane or thread and the iteration summed into one index); the
terms show the fold when they know every value in them, and a map through a
value loaded per thread or per lane (a jagged tile's rows begin at an offset
loaded per row) is not modelled.  A packet (a vector load hoisted above a
V-loop, the flush of its stores) is every one of its elements: its term
along the vectorized dimension is the per-thread base's plus the V-loop's
variable, so the packet's later elements are compared as much as its first.

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
it is rejected rather than barriered (only an ``hl.if`` whose condition
``cute/block_uniform.py`` proves the same for the whole CTA is not); the
chunks of a looped reduction tell which of two accesses meeting in two
chunks runs first only through terms affine in variables of known ranges
(``AddressMaps.apart_in_later_iterations``), so a load meeting only a
later chunk's store through other terms is rejected.
Known gaps (programs accepted whose nest is not the program): a store
through a value loaded per thread or per lane (``x[base[t0] +
tk.begin]``, a jagged tile's rows beginning at
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
import dataclasses
from itertools import starmap
import logging
import operator
from typing import TYPE_CHECKING
from typing import cast

import sympy

from .... import exc
from ...ast_extension import expr_from_string
from ...ast_extension import statement_from_string
from ...ast_read_writes import HELION_BLOCK_UNIFORM_ATTR
from ...ast_read_writes import HELION_RAW_BARRIER_TENSORS_ATTR
from ..address_maps import AddressMaps
from ..address_maps import Load
from ..address_maps import _tighter
from ..address_maps import flat_terms
from ..address_maps import iteration_symbol
from ..address_maps import loop_symbol
from ..address_maps import thread_symbol
from ..address_maps import uniform_symbol
from ..address_maps import whole_quantities
from .markers import _HELION_LANE_REDUCE_MARKER
from .markers import _is_lane_reduce_marker_assign
from .statements import LOGGER_NAME
from .statements import LanePlacement
from .statements import LaneScope
from .statements import _Access
from .statements import _access_addresses
from .statements import _address_terms
from .statements import _analyze
from .statements import _conjuncts
from .statements import _defining_load
from .statements import _dependent_names
from .statements import _disjoint
from .statements import _guarded_statements
from .statements import _is_device_loop
from .statements import _names_mentioned
from .statements import _packets
from .statements import _propagate_dependence
from .statements import _propagate_thread_axes
from .statements import _scope_packet
from .statements import _Shifts
from .statements import _simple_statements
from .statements import _Statement
from .statements import _structural_markers
from .statements import _thread_axis_reads

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import Mapping
    from collections.abc import Sequence

    from ..address_maps import Bound

log = logging.getLogger(LOGGER_NAME)


_BARRIER_CALL = "cute.arch.sync_threads"


def is_barrier(node: ast.AST) -> bool:
    """Whether ``node`` is a ``cute.arch.sync_threads()`` statement."""
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == _BARRIER_CALL
    )


def _rely(block: _Block, positions: Sequence[int]) -> bool:
    """Whether a barrier at one of ``positions`` in ``block`` orders a pair.

    When every one of them is a load's read-after-write barrier, tagged with
    the tensors whose stores it orders (``HELION_RAW_BARRIER_TENSORS_ATTR``),
    the first loses its tag: it orders this pair too, so no later pass may
    treat it as ordering only those stores.
    """
    if not positions:
        return False
    items = cast("list[ast.AST]", block.items)
    tagged = [
        position
        for position in positions
        if position in block.existing
        and getattr(items[position], HELION_RAW_BARRIER_TENSORS_ATTR, None) is not None
    ]
    if len(tagged) == len(positions):
        delattr(items[tagged[0]], HELION_RAW_BARRIER_TENSORS_ATTR)
    return True


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


def _iteration_buffers(nodes: Iterable[ast.AST]) -> set[str]:
    """The names ``nodes`` bind to shared memory they also allocate
    (``cute.make_tensor`` of a ``cute.arch.alloc_smem`` pointer bound among
    them, as a rebinding's exchange buffer is): storage of one iteration of
    the loop whose body they are, which holds nothing from one iteration to
    the next."""
    pointers: set[str] = set()
    names: set[str] = set()
    for node in nodes:
        for child in ast.walk(node):
            if not (
                isinstance(child, ast.Assign)
                and len(child.targets) == 1
                and isinstance(child.targets[0], ast.Name)
                and isinstance(child.value, ast.Call)
            ):
                continue
            call = child.value
            if ast.unparse(call.func) == "cute.arch.alloc_smem":
                pointers.add(child.targets[0].id)
            elif ast.unparse(call.func) == "cute.make_tensor" and (
                call.args
                and isinstance(pointer := call.args[0], ast.Name)
                and pointer.id in pointers
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
    barrier can be placed in it (nor in anything nested in it), unless the
    branch is marked uniform across the CTA (``HELION_BLOCK_UNIFORM_ATTR``).
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


def _first_conjuncts(scope: LaneScope, iteration: frozenset[str]) -> frozenset[str]:
    """The conjuncts of the scope's first-iteration predicate reading ``iteration``."""
    if not scope.first_lane:
        return frozenset()
    test = expr_from_string(scope.first_lane)
    assert isinstance(test, ast.expr)
    return frozenset(
        conjunct
        for conjunct in _conjuncts(test)
        if _names_mentioned(expr_from_string(conjunct), {}) & iteration
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
    inside the lane nest without passing through it), unless the branch's
    condition is proven the same for every thread of the CTA
    (``HELION_BLOCK_UNIFORM_ATTR``): its sides then take the divergence of
    the enclosing list, and a barrier placed in one is inserted into the
    branch itself.  Inside a lane loop the sides stay in one iteration of
    it, like the loop's own list: the exactness check reads their
    lane-invariant statements (``_inexact_repeated_branch``) and the pairs
    across the lanes are paired below as anywhere in the loop.  A device
    loop, a guard or any other statement is one leaf.  A vectorized
    loop's wrapper block is the body of its lane loop, whose iteration
    changes every coordinate of the scope and whose variable is the lane
    variable; the placement's items are the body of its V-loop, whose
    iteration changes the coordinates its attached statements do not
    read (the V-loop variable, its variable), or of the plain lane loop.
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
            uniform = _uniform_branch(item)
            branches = [
                (
                    item.body,
                    _conjuncts(item.test) if isinstance(item, ast.If) else frozenset(),
                ),
                *([(item.orelse, frozenset())] if item.orelse else []),
            ]
            for branch, test in branches:
                _leaves(
                    cast("list[ast.AST | LanePlacement]", branch)
                    if uniform
                    else list(branch),
                    scopes,
                    renames,
                    here,
                    blocks,
                    leaves,
                    divergent=divergent or not uniform,
                    conjuncts=test,
                )
            continue
        if not isinstance(item, LanePlacement):
            if is_barrier(item):
                block.existing.add(index)
            leaves.append(_Leaf(_analyze(-1, item, renames), here))
            continue
        scope = scopes[item.lane_var]
        attached_reads: set[str] = set()
        for node in scope.attached:
            attached_reads |= _names_mentioned(node, renames)
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


def _uniform_branch(item: object) -> bool:
    """Whether ``item`` is a branch every thread of the CTA takes alike."""
    return isinstance(item, ast.If) and getattr(item, HELION_BLOCK_UNIFORM_ATTR, False)


def _exclusive_sides(first: _Leaf, second: _Leaf) -> bool:
    """Whether the leaves sit on the two (non-divergent) sides of a uniform branch.

    The whole CTA takes one side of such a branch, so the two never run in
    one pass of the statements around it.
    """
    parent: tuple[_Block, int] | None = None
    for (block, index), (other, other_index) in zip(
        first.path, second.path, strict=False
    ):
        if block is not other:
            assert parent is not None
            items = parent[0].items
            return (
                not block.divergent
                and items is not None
                and _uniform_branch(items[parent[1]])
            )
        if index != other_index:
            return False
        parent = (block, index)
    return False


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
    lane_variables: frozenset[str] = frozenset(),
    packets_before: Sequence[ast.AST] = (),
    packets_after: Sequence[ast.AST] = (),
    packet_calls: Mapping[int, tuple[str, str]] | None = None,
    chunk_shifts: _Shifts | None = None,
) -> None:
    """Separate with block-wide barriers the tensor accesses threads could race on.

    ``items`` is the nest about to be emitted (placement items and
    ``LanePlacement`` loops, or the body statements themselves when no lane
    loop is live); ``scopes`` describes the live loops with their attached
    statements and trip counts; ``axis_sizes`` gives the launch thread count
    along each thread axis the body may address; ``definitions`` are the
    per-thread index and mask definitions around the body (the lane setup,
    the thread axes without a lane loop, the vec wrappers' lane bases, the
    statements the per-thread loops among ``loop_headers`` run before the
    body) and ``masks`` the tile masks among them; ``constants`` the values
    of the block-size constants the loop bounds inside the body name;
    ``loop_headers`` the ``for`` statements of the device loop whose body
    the nest is and of the per-thread loops between it and the nest that
    the nest does not hold (a looped reduction's lane loop and V-loop,
    whose targets are ``lane_variables``), whose bounds model their
    variables across iterations, and of the loops around them (the
    enclosing device loops and lane loops: their variables are one value
    each throughout the body, nonnegative and below their bounds,
    ``AddressMaps``).  Two statements accessing one tensor, at least one
    writing it, whose threads may differ along an axis shared by several
    threads (the address mappings of the two accesses do not force the
    coordinate equal, ``AddressMaps.same_thread``, the indices the tile
    masks around each access bound counting as bounded,
    ``AddressMaps.bounds``: a uniform address, a gathered one, a
    neighbour's or a permuted element) get a barrier between them at their
    nearest common statement list unless ``_races`` proves the order
    immaterial or the recorded regions of their accesses are apart
    (``_disjoint``); a barrier
    already there counts, and a load's read-after-write barrier counted on
    loses its tag (``HELION_RAW_BARRIER_TENSORS_ATTR``), since it then
    orders more than its load's stores.  Modifies ``items`` in
    place: barrier statements
    are inserted among placement items, and positions among a vectorized
    loop's wrapper statements are recorded on its ``LanePlacement.barriers``.
    A pair whose barrier would sit in control flow a thread may skip, where
    a block-wide barrier deadlocks (the body of a branch or while loop among
    the items, or the whole nest without ``allow_barriers``), is rejected
    with ``BackendUnsupported`` instead; a branch marked uniform across the
    CTA is no such control flow, and a pair on its two sides never meets.
    So is any pair needing a barrier in a body holding a ``_helion_*``
    placeholder (``_compiler_markers``), which a later pass rewrites.

    Every loop of the nest runs its body once per iteration on every thread
    with nothing between the iterations.  The device loop whose body the
    nest is (``loop_variables`` names what it redefines per iteration: its
    ``for`` targets and those of the per-thread loops between it and the
    nest, whose iterations count as its own, run in program order inside
    each of its iterations; ``loop_body`` is the statement list one
    iteration runs: ``items``, or the innermost loop's) runs its iterations
    one after the other in the program: two statements of its body, in
    either emission order and a statement with itself included on a tensor
    it both reads and writes (a device loop nested in the body holds a
    uniform read and a per-thread store of its own; its unknown effects,
    the tensors inside calls the pass cannot see through, are not ordered
    against its own next instance) or only stores to through a map the
    terms know whole (which folds between the iterations when the thread
    and the iteration sum into one index), racing as one iteration's and a
    later one's, need a barrier after the first within its iteration or
    before the second within its own; when none of the placed ones does,
    one is added at the end of the body (or right after the first
    statement, when that sits among a vectorized loop's wrapper statements
    past the body).  Only the recorded regions of the accesses exempt such
    a pair, when they cannot meet in any later iteration (``loop_shifts``
    maps the loop's begin symbols to their step, or to None when the next
    iteration's begins are unrelated, as when per-thread loops make the
    next iteration another lane of the same tile: ``_disjoint``), or the
    address mappings, when they force the thread equal with the loop's
    variables taken as differing between the two iterations; a uniform
    access whose address moves with the iteration may still meet the
    previous iteration's per-thread stores.  The per-thread loops'
    iterations are lanes of the tile statements, though: a pair that may
    meet in two of them within one iteration of the device loop (their
    equal terms do not force every variable of ``lane_variables`` equal),
    one side writing, is rejected with ``BackendUnsupported``, since no
    barrier orders a thread's own lanes (a statement meets itself there on
    a tensor it both reads and writes; two stores whose addresses ignore
    the lanes are not paired).  Their packets are accesses of every element
    of a lane's vector, paired by that rule too: those they load before the
    body (``packets_before``) and flush after it (``packets_after``), with
    ``packet_calls`` giving each packet call's V-loop variable and
    per-thread base (``_packets``).  With ``chunk_shifts`` (the step of each
    tile begin per chunk) the device loop walks the chunks of a looped
    reduction, whose tile statements the program runs over every chunk
    before the next one (a load of every element before a store of any): a
    pair that may meet in two chunks (its regions are not apart and its
    equal terms do not force the chunk equal), one side writing, is
    rejected with ``BackendUnsupported`` too, unless the loop runs one
    chunk (a cluster split's slice) or the program's earlier statement,
    run in a later chunk, never reaches an element the later one reached
    in an earlier chunk (``AddressMaps.apart_in_later_iterations``: a load
    of ``out[t, 64:]`` beside a store of ``out[t, :n]``).  A barrier orders
    threads, not chunks, whatever the addresses: ``out[t, 64:] = out[t,
    :n] + 1`` stores in one chunk what the next one loads.
    A lane
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
    # A later pass rewrites a body holding a placeholder (a reduction's
    # ``_helion_lane_reduce``, a collective matmul's
    # ``_helion_pending_collective_mma``), moving its statements between the
    # passes it builds and refusing a barrier among the ones it stages: the
    # accesses are checked as for any body, but a barrier they need rejects it.
    placeholders = _structural_markers([leaf.statement.node for leaf in leaves])
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
    # The split combines a reduction over its threads and gives every one of
    # them the result (``_reduced_thread_axes``): two stores of one tensor
    # whose addresses and values do not vary along the axes their threads
    # differ on write the same sequence on each thread, ending with the
    # program's value whatever the threads' order.
    reduced_axes = (
        _reduced_thread_axes(statements, parts, everything)
        if _HELION_LANE_REDUCE_MARKER in placeholders
        else None
    )
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
        # The flushes after the body read its definitions and define nothing.
        definitions=[*definitions, *packets_after],
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
                if base in _names_mentioned(term, rename_groups)
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
                    _names_mentioned(term, rename_groups) - {name}
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
        if (
            reduced_axes is not None
            and tensor not in first.tensors_read | second.tensors_read
            and not reduced_axes[id(second)] & differing
            and (
                not reduced_axes[id(first)] & differing
                or leader_writes_after(first, second, tensor, differing, varying)
            )
        ):
            return False
        return bool(differing) and _races(first, second, differing, [tensor])

    def leader_writes_after(
        first: _Statement,
        second: _Statement,
        tensor: str,
        axes: set[int],
        varying: Iterable[str],
    ) -> bool:
        """Whether ``first``'s write of each element ``second`` writes is the leader's, along ``axes``.

        ``second`` stores one value along ``axes``; the reduction split either
        has every thread store it, each thread's last write, or guards it to
        the leader of the axes (coordinate 0), which then follows its own
        write of the element in ``first``.
        """
        mine = address_maps.get(id(first), {}).get(tensor)
        theirs = address_maps.get(id(second), {}).get(tensor)
        if mine is None or theirs is None or tensor in first.opaque | second.opaque:
            return False
        leader = maps.bounds(
            second.node,
            [f"cute.arch.thread_idx()[{axis}] == 0" for axis in sorted(axes)],
        )
        second_bounds = dict(bounds[id(second)][tensor])
        for symbol, bound in leader.items():
            second_bounds[symbol] = _tighter(second_bounds.get(symbol, bound), bound)
        # The leader's coordinates, zero by their bounds, join the address
        # where ``first``'s does, so the proof relates them to its thread.
        theirs = [list(access) for access in theirs]
        for axis in axes:
            symbol = thread_symbol(axis)
            positions = {
                index
                for access in mine
                for index, term in enumerate(access)
                if term is not None and symbol in term.free_symbols
            }
            if len(positions) != 1:
                return False
            (position,) = positions
            for access in theirs:
                term = access[position] if position < len(access) else None
                if term is None:
                    return False
                access[position] = sympy.Add(term, symbol)
        equal = forced_equal(
            mine,
            theirs,
            maps.pair_ranges(bounds[id(first)][tensor], second_bounds),
            varying,
        )
        return all(thread_symbol(axis) in equal for axis in axes)

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
            if tensors and not _exclusive_sides(first, second):
                block, after, before = _common_block(first, second)
                if _rely(
                    block,
                    [
                        barrier
                        for barrier in block.existing
                        if after < barrier <= before
                    ],
                ):
                    # Ordered already, by a barrier the lowering emitted
                    # itself (a scan's, between its shared-memory phases, or
                    # a load's read-after-write barrier).
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

    def shared_pairs(
        statements: Sequence[_Statement],
        models: Sequence[Mapping[str, list[list[sympy.Expr | None]] | None]],
        own: Sequence[frozenset[str]],
        symbols: set[sympy.Symbol],
        *,
        lane_level: bool = False,
    ) -> Iterator[tuple[int, int, str]]:
        """``(earlier, later, tensor)``: the positions of two of ``statements``, or of one twice, and a tensor outside ``private`` they both access, which two of their instances may meet on.

        A statement meets itself on the tensors ``own`` gives at its
        position.  Two stores whose addresses (``models``) read none of
        ``symbols`` (what tells the instances apart) nor an unknown value
        re-apply the same values in program order whichever instance runs
        first, and are not paired.  ``lane_level`` leaves out the tensors
        whose accesses across a lane loop's lanes the emitter ordered
        (``lane_ordered``).
        """
        for earlier, first in enumerate(statements):
            for later in range(earlier, len(statements)):
                second = statements[later]
                tensors = (first.tensors & second.tensors) - private
                if lane_level:
                    tensors -= first.lane_ordered & second.lane_ordered
                if later == earlier:
                    tensors &= own[earlier]
                for tensor in sorted(tensors):
                    if not (
                        stores_only(first, tensor)
                        and stores_only(second, tensor)
                        and invariant(models[earlier].get(tensor), symbols)
                        and invariant(models[later].get(tensor), symbols)
                    ):
                        yield earlier, later, tensor

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
        own = list(starmap(own_pairs, zip(inner, models, strict=True)))
        for position, later, tensor in shared_pairs(inner, models, own, level_symbols):
            first, second = inner[position], inner[later]
            if not conflicting(first, second, tensor, None):
                continue
            if tensor in first.opaque or tensor in second.opaque:
                equal: frozenset[sympy.Symbol] = frozenset()
            else:
                equal = forced_equal(
                    models[position].get(tensor),
                    models[later].get(tensor),
                    maps.pair_ranges(inner_bounds[position], inner_bounds[later]),
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
            raise exc.BackendUnsupported(
                "cute",
                f"a loop repeated whole once per {', '.join(sorted(variables))} "
                f"accesses {tensor} in an order the lanes' passes interleave: "
                f"{ast.unparse(first.node).splitlines()[0][:80]!r} and "
                f"{ast.unparse(second.node).splitlines()[0][:80]!r} are not "
                "confined to one lane, which no barrier orders as the program "
                "does (the loop would have to be split)",
            )

    # The per-thread loops among the device loop's (``lane_variables``: a
    # looped reduction's lane loop and V-loop) run the lanes of the tile
    # statements one after another on each thread, inside one iteration of
    # the device loop, while the program runs every lane of a statement
    # before any of the next one.  Two accesses of one tensor, one writing,
    # that may meet in two of those lanes (their equal terms do not force
    # every lane variable equal, with the device loop's own variables one
    # value) are observed out of program order on whichever threads run
    # them: a later lane's gather of an element an earlier lane stored reads
    # the new value, and no barrier orders a thread's own lanes.  Such a
    # pair is rejected, as across the lanes of a lane loop of the nest; a
    # statement meets itself on a tensor it both reads and writes, and two
    # stores whose addresses ignore the lanes (and every value loaded per
    # lane) re-apply their values in program order.  The packets these loops
    # load before the body and flush after it are accesses of every element
    # of a lane's vector, in program order around the body's statements.
    # A looped reduction's chunks (``chunked``) are the device loop's
    # iterations, but of one tile statement too: the program runs a
    # statement over every chunk before the next statement, so a pair that
    # may meet in two chunks, one writing, is observed out of program order
    # whatever the barriers (a later chunk's load of an element an earlier
    # chunk stored, gathered or at a shifted address), and is rejected as
    # well.
    sequence = [
        *(_analyze(-1, node, rename_groups) for node in packets_before),
        *(leaf.statement for leaf in leaves),
        *(_analyze(-1, node, rename_groups) for node in packets_after),
    ]
    for statement in sequence:
        if id(statement) in address_maps:
            continue
        address_maps[id(statement)] = modelled(
            statement.node, statement.node, packet_calls or {}, None, frozenset()
        )
        bounds[id(statement)] = {
            canonical(tensor): maps.bounds(statement.node, ())
            for tensor in _access_addresses(statement.node, packet_calls or {})
        }

    def candidates(
        variables: set[str],
    ) -> Iterator[tuple[_Statement, _Statement, str, set[str]]]:
        """The ``shared_pairs`` of ``sequence`` the instances of ``variables``
        (lanes, chunks) may order wrongly, the later statement first, with
        the names varying with the variables.  A statement meets itself on a
        tensor it reads and writes (its stores in two lanes are one tile
        statement's), and two stores pass only when their addresses ignore
        every value the variables define too (a value loaded per lane is
        another value in another lane)."""
        varying = variables | _dependent_names(
            everything,
            (bool(statement.reads & variables) for statement in everything),
            masks,
        )
        for earlier, later, tensor in shared_pairs(
            sequence,
            [address_maps[id(statement)] for statement in sequence],
            [
                (statement.tensors_read | statement.opaque)
                & (statement.tensors_written | statement.opaque)
                for statement in sequence
            ],
            {uniform_symbol(name) for name in varying},
            lane_level=True,
        ):
            yield sequence[later], sequence[earlier], tensor, varying

    def iterations_of(variables: set[str]) -> set[sympy.Symbol]:
        """The iterations of ``variables``, all of them modelled, or none."""
        iterations = {
            iteration_symbol(name)
            for name in variables
            if uniform_symbol(name) in maps.iteration_values
        }
        return iterations if len(iterations) == len(variables) else set()

    def out_of_order(tensor: str, where: str) -> exc.BackendUnsupported:
        return exc.BackendUnsupported(
            "cute",
            f"an access of {tensor} (gathered, or another lane's or chunk's "
            f"element) may meet another access of it in {where} of a looped "
            "reduction, which no barrier orders as the program does (its load "
            "and store would have to run in separate sweeps)",
        )

    def later_chunks_apart(
        first: _Statement, second: _Statement, tensor: str, varying: set[str]
    ) -> bool:
        """Whether ``second``'s accesses of ``tensor`` in a later chunk never
        reach an element ``first``'s reach in an earlier one
        (``AddressMaps.apart_in_later_iterations``)."""
        if tensor in first.opaque or tensor in second.opaque:
            return False
        mine = address_maps.get(id(first), {}).get(tensor)
        theirs = address_maps.get(id(second), {}).get(tensor)
        if mine is None or theirs is None:
            return False
        ranges = maps.pair_ranges(bounds[id(first)][tensor], bounds[id(second)][tensor])
        return all(
            maps.apart_in_later_iterations(
                access, other, loops=chunks, ranges=ranges, varying=varying
            )
            for access in mine
            for other in theirs
        )

    lanes = {canonical(name) for name in lane_variables}
    chunks = {canonical(name) for name in loop_variables - lane_variables}
    if lanes:
        lane_iterations = iterations_of(lanes)
        for first, second, tensor, varying in candidates(lanes):
            if not conflicting(first, second, tensor, {}):
                continue
            if lane_iterations and lane_iterations <= same_thread(
                first, second, tensor, varying
            ):
                continue
            raise out_of_order(
                tensor, f"two lanes ({', '.join(sorted(lanes))}) of one chunk"
            )
    walks = [
        maps.trip_count(loop)
        for loop in loop_headers
        if isinstance(loop.target, ast.Name) and canonical(loop.target.id) in chunks
    ]
    one_chunk = bool(walks) and all(count is not None and count <= 1 for count in walks)
    if chunk_shifts is not None and not one_chunk:
        chunk_iterations = iterations_of(chunks)
        # A buffer each chunk allocates for itself (a rebinding's exchange
        # through shared memory) holds nothing from one chunk to the next:
        # the program's order over the chunks says nothing about it, and the
        # barriers order its uses within and between them.
        buffers = {
            canonical(name)
            for name in _iteration_buffers(statement.node for statement in sequence)
        }
        for first, second, tensor, varying in candidates(chunks):
            if tensor in buffers or not conflicting(
                first, second, tensor, chunk_shifts
            ):
                continue
            # The two chunks' instances may be any of their lanes.
            varying = varying | lanes
            if chunk_iterations and chunk_iterations <= same_thread(
                first, second, tensor, varying
            ):
                continue
            # The earlier chunk runs the program's later statement first.
            if later_chunks_apart(first, second, tensor, varying):
                continue
            raise out_of_order(tensor, "two chunks")

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
        if not sequential and reduced_axes is not None:
            # The reduction split rebuilds the lane loops into its passes and
            # proves their order itself (``_lane_split_reorders_aliasing_memory``).
            continue
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
                    ordered = (
                        first.statement.lane_ordered & second.statement.lane_ordered
                    )
                    tensors = {
                        tensor
                        for tensor in shared_tensors - ordered
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
            if _rely(
                block, [barrier for barrier in barriers_of(block) if barrier > index]
            ):
                return True
        for block, index in _below(second.path, level):
            if block.divergent:
                break
            if _rely(
                block, [barrier for barrier in barriers_of(block) if barrier <= index]
            ):
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
    if placeholders:
        if any(placed[id(block)] for block in blocks):
            raise exc.BackendUnsupported(
                "cute",
                "threads sharing a tile axis race on a tensor in a body a later "
                f"pass rewrites ({', '.join(sorted(placeholders))}), where the "
                "barrier ordering them cannot be placed",
            )
        return
    for block in blocks:
        for position in sorted(placed[id(block)]):
            log.debug("thread barrier placed %s", _describe(block, position))
        block.insert(placed[id(block)])


_THREAD_REDUCTION_HELPERS = ("_cute_grouped_reduce", "warp_reduction")


def _combined_reduction(node: ast.AST) -> bool:
    """Whether ``node`` assigns a reduction combined over its threads, and nothing else.

    A ``_helion_lane_reduce`` marker, or a grouped or warp reduction helper,
    either possibly under a cast: each gives every thread of its group the
    group's result.
    """
    if _is_lane_reduce_marker_assign(node) is not None:
        return True
    if not isinstance(node, ast.Assign):
        return False
    value = node.value
    if (
        isinstance(value, ast.Call)
        and ast.unparse(value.func).startswith("cutlass.")
        and len(value.args) == 1
        and not value.keywords
    ):
        value = value.args[0]
    return isinstance(value, ast.Call) and (
        ast.unparse(value.func).rsplit(".", 1)[-1].startswith(_THREAD_REDUCTION_HELPERS)
    )


def _reduced_thread_axes(
    statements: Sequence[_Statement],
    parts: Sequence[tuple[_Statement, list[_Statement]]],
    everything: Sequence[_Statement],
) -> dict[int, set[int]]:
    """Per statement (by ``id``), the thread axes its values vary along once its reductions are combined.

    For ``add_thread_barriers`` in a body the reduction split rewrites: the
    result of a reduction combined over its threads (``_combined_reduction``)
    is one value on every thread of the reduction's group, and a store's
    index covers every axis its value still varies along, so the result is
    taken to read no thread coordinate.  Unlike ``_propagate_thread_axes``
    the masks count: a thread a mask skips writes nothing.  A compound
    statement takes its simple statements' axes.
    """
    cut = [
        dataclasses.replace(statement, reads=frozenset())
        if _combined_reduction(statement.node)
        else statement
        for statement in everything
    ]
    seeds = [_thread_axis_reads(statement.node) for statement in everything]
    axes = {
        id(statement): dependence
        for statement, dependence in zip(
            everything, _propagate_dependence(cut, seeds, frozenset()), strict=True
        )
    }
    for statement, analyzed in parts:
        axes[id(statement)] = set().union(*(axes[id(piece)] for piece in analyzed))
    for statement in statements:
        axes.setdefault(id(statement), set())
    return axes


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
