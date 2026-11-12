"""Per-statement analysis shared by the lane-loop placement and the barrier pass.

``_analyze`` reads one statement of a lane-loop nest into a ``_Statement``:
the names it reads and writes, the tensors it loads, stores or updates
atomically with their recorded regions, the lane coordinates it depends on,
and whether running it once outside a lane loop is the tile program's
meaning.  ``placement.py`` distributes the statements over the lane loops
from these facts and ``barriers.py`` separates the accesses threads could
reorder; ``LaneScope`` and ``LanePlacement`` are the nest they both work
on.  The three modules log under one name (``LOGGER_NAME``).

Memory dependences key on the generated tensor names, as the rest of the
lane-loop lowering does.  The rename groups map a kernel argument viewing
another's storage (a host-side ``x.view(...)``) to that argument's name;
arguments that alias only at runtime (one tensor passed twice) are not
ordered against each other.  A tensor the body builds itself (the
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
"""

from __future__ import annotations

import ast
import collections
from collections.abc import Mapping
import dataclasses
from typing import TYPE_CHECKING
from typing import TypeVar

import sympy

from ...ast_read_writes import HELION_ACCESS_REGIONS_ATTR
from ...ast_read_writes import HELION_ATOMIC_UNIFORM_LANES_ATTR
from ...ast_read_writes import HELION_FENCE_BARRIER_ATTR
from ...ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from ...ast_read_writes import HELION_LANE_ORDERED_ATTR
from ...ast_read_writes import ReadWrites
from .._ast_pass_utils import _is_proven_relocatable_call
from .._ast_pass_utils import exposed_reads
from .._ast_pass_utils import is_vloop
from ..access_regions import access_address
from ..address_maps import Load
from ..cache_policy_loads import _CUTE_CACHE_LOAD_HELPER_NAMES
from .markers import _find_lane_reduce_call
from .markers import _is_lane_reduce_marker_assign
from .owner_guards import _VECTOR_ATOMIC_HELPERS
from .owner_guards import _is_atomic_helper
from .owner_guards import _memory_write_calls

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import Sequence


LOGGER_NAME = "helion._compiler.cute.lane_loop_distribution"

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


def _compiler_markers(body: list[ast.AST]) -> set[str]:
    """The ``_helion_*`` placeholders a later pass still has to expand."""
    return {
        node.func.id
        for statement in body
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id.startswith("_helion_")
    }


# The persistent-reduction vector markers wrap one scalar access each (the
# load's original expression, the store's pointer and value); the pass
# expanding them (``cute/persistent_branch_vec.py``) walks whatever nest it is
# given and places each packet beside the constexpr lane loop holding its
# site, so a body holding only these markers is placed like a scalar body.
_PLACED_MARKERS = frozenset({_PERSISTENT_VEC_LOAD, _PERSISTENT_VEC_STORE})


def _structural_markers(body: list[ast.AST]) -> set[str]:
    """The placeholders whose expanding pass needs the full lane nest."""
    return _compiler_markers(body) - _PLACED_MARKERS


def contains_compiler_marker(body: list[ast.AST]) -> bool:
    """Whether a later pass still has to expand a ``_helion_*`` placeholder."""
    return bool(_compiler_markers(body))


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
    """``cute.arch.atomic_*(ptr, ...)`` or an atomic called by bare name: a
    generated atomic helper or an ``atomic_*`` function
    (``owner_guards._is_atomic_helper``).  A pointer's ``atomic_*`` method is
    not one, unlike in ``owner_guards._is_atomic_call``."""
    if isinstance(call.func, ast.Attribute):
        return call.func.attr.startswith("atomic_") and (
            ast.unparse(call.func.value) == "cute.arch"
        )
    return isinstance(call.func, ast.Name) and _is_atomic_helper(call.func.id)


def _uniform_lanes_of(call: ast.Call) -> frozenset[str]:
    """The lane loops the atomic call is recorded as uniform along.

    ``cute/atomic_ops.py`` records them on the calls it guards; an atomic
    another lowering emits carries no record and is uniform along no loop.
    """
    return frozenset(getattr(call, HELION_ATOMIC_UNIFORM_LANES_ATTR, ()))


def _is_relaxed_atomic(call: ast.Call) -> bool:
    if isinstance(call.func, ast.Name) and call.func.id in _VECTOR_ATOMIC_HELPERS:
        return True
    return any(
        keyword.arg == "sem"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value == "relaxed"
        for keyword in call.keywords
    )


def _atomic_addresses_a_tensor(call: ast.Call) -> bool:
    """Whether the atomic's pointer operand (its first argument) names its tensor."""
    return bool(call.args) and bool(_addressed_tensors(call.args[0]))


def _calls_are_movable(
    node: ast.AST, *, also: Callable[[ast.Call], bool] | None = None
) -> bool:
    return all(
        _is_plain_store(call)
        or _is_plain_load(call)
        or _is_list_append(call)
        or (also is not None and also(call))
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


def _is_movable(
    node: ast.AST, *, also: Callable[[ast.Call], bool] | None = None
) -> bool:
    """Whether running ``node`` once instead of once per lane preserves it.

    Assignments and expression statements made of proven pure calls (loads
    included), ordinary stores and store buffer appends qualify, as do
    ``range`` loops and branches built only from them.  Anything else pins
    the statement.  The calls ``also`` accepts qualify too: with
    ``_is_atomic_call`` such a statement is placed by its reads and its
    tensor and never repeated.
    """
    if isinstance(node, ast.For):
        return (
            not node.orelse
            and _is_movable_iterator(node.iter)
            and all(_is_movable(child, also=also) for child in node.body)
        )
    if isinstance(node, ast.If):
        return _calls_are_movable(node.test) and all(
            _is_movable(child, also=also) for child in [*node.body, *node.orelse]
        )
    if isinstance(node, ast.Pass):
        return True
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr)):
        return _calls_are_movable(node, also=also)
    return False


def runs_once_outside_lane(node: ast.AST, lane_var: str) -> bool:
    """Whether ``node`` keeps its effect run once instead of once per lane of ``lane_var``.

    For a statement that reads none of the loop's coordinates
    (``_sink_lane_loop_into_branches``): the statements ``_is_movable``
    accepts, where an atomic recorded uniform along the loop and the
    reduction marker of another lane loop qualify too.  An atomic that is
    not uniform along it adds once per lane, and the loop's own marker
    reduces over its lanes.
    """
    foreign_markers: set[int] = set()
    for child in ast.walk(node):
        marker = _is_lane_reduce_marker_assign(child)
        if marker is None:
            continue
        if marker.owner_lane in (None, lane_var):
            return False
        assert isinstance(child, ast.Assign)
        foreign_markers.add(id(_find_lane_reduce_call(child.value)))

    def also(call: ast.Call) -> bool:
        if _is_atomic_call(call):
            return lane_var in _uniform_lanes_of(call)
        return id(call) in foreign_markers

    return _is_movable(node, also=also)


def _pointer_strategy_tensor(node: ast.AST) -> str | None:
    """The tensor an access of the shared pointer indexing strategy
    (``indexing_strategy.py``) addresses: ``tl.load(x + offset, mask, ...)``,
    ``tl.store(x + offset, value, mask)`` and ``tl.atomic_add(x + offset,
    value, ...)`` address ``x``, the base of the sum in their first argument.
    None for any other node."""
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and (
            node.func.attr in ("load", "store") or node.func.attr.startswith("atomic_")
        )
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "tl"
        and node.args
    ):
        return None
    base = node.args[0]
    while isinstance(base, ast.BinOp) and isinstance(base.op, ast.Add):
        base = base.left
    return base.id if isinstance(base, ast.Name) else None


def _mentions(
    node: ast.AST, skip: Callable[[ast.AST], bool] | None = None
) -> Iterator[str]:
    """Generated tensor names addressed in ``node``, one per mention.

    ``t.iterator``, ``t.__setitem__``, ``t[...]`` and the pointer indexing
    strategy's accesses (``tl.load(t + offset, ...)``,
    ``_pointer_strategy_tensor``) address ``t``.  A metadata query of the
    tensor or of a slice of it (``t[...].shape``, ``t[...].layout``,
    ``t.iterator.alignment``: ``_METADATA_ATTRS``) reads no element and
    addresses nothing, though an index inside the slice may (a load).
    ``skip`` excludes subtrees.  Aliasing kernel arguments have distinct
    names and are not related here.
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
        elif (tensor := _pointer_strategy_tensor(child)) is not None:
            yield tensor
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
        tensor = _pointer_strategy_tensor(call)
        if tensor is not None and ast.unparse(call.func) == "tl.store":
            written.add(tensor)
            read[tensor] -= 1
            continue
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


def _is_device_loop(node: ast.AST) -> bool:
    """A loop of the tile program (a ``hl.tile`` / ``hl.range`` nest's ``for``): not a lane loop, not a V-loop."""
    return (
        isinstance(node, ast.For)
        and getattr(node, HELION_LANE_LOOP_VAR_ATTR, None) is None
        and not is_vloop(node)
    )


def _is_packet_call(call: ast.Call) -> bool:
    """A vector load or store of consecutive elements through its pointer operand."""
    if isinstance(call.func, ast.Name):
        return (
            call.func.id in _PLAIN_STORE_HELPERS
            or call.func.id in _PLAIN_LOAD_HELPERS
            or call.func.id in _VECTOR_ATOMIC_HELPERS
        )
    return ast.unparse(call.func) in ("cute.arch.load", "cute.arch.store") and any(
        isinstance(argument, ast.Call)
        and ast.unparse(argument.func) == "ir.VectorType.get"
        for argument in call.args
    )


def _packets(node: ast.AST) -> dict[int, tuple[str, str]]:
    """The packets of the vec wrappers inside ``node``: ``(V-loop variable, per-thread base)`` by call id.

    A vec wrapper is a loop whose body holds one V-loop (``is_vloop``)
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
        vloops = [statement for statement in loop.body if is_vloop(statement)]
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
                bases = _names_mentioned(address, {}) & defined
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
    ``t.__setitem__(index, value)`` mention's the index, a ``t[index]``
    mention's the index and a pointer indexing strategy access's
    (``tl.load(t + offset, ...)``) the pointer ``t + offset``, so every
    mention ``_tensor_mentions`` counts has one address here, with the
    elements it covers (``_Access``; ``packets`` are the wrappers' packets
    in view, ``_packets``).
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
        elif (tensor := _pointer_strategy_tensor(child)) is not None:
            assert isinstance(child, ast.Call)
            addresses[tensor].append(_Access(child.args[0], True))
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
        if isinstance(call, ast.Call) and _is_packet_call(call):
            # A vector atomic flush through ``p.llvm_ptr``: its packet.
            vector, base = packets.get(id(call), (None, None))
            return _Access(address, False, vector, base)
        if isinstance(call, ast.Call) and _is_atomic_call(call):
            # ``cute.arch.atomic_add(p.llvm_ptr, ...)``: one element.
            return _Access(address, True)
        return _Access(address, False)
    if isinstance(owner, ast.Call):
        if _is_atomic_call(owner):
            return _Access(address, True)
        if _is_persistent_vec_access(
            owner, _PERSISTENT_VEC_LOAD
        ) or _is_persistent_vec_access(owner, _PERSISTENT_VEC_STORE):
            # A persistent-reduction marker stands for its lane's scalar
            # access; the later pass makes the lanes' exact fragment one
            # vector transaction of the same thread.
            return _Access(address, True)
        if _is_packet_call(owner):
            vector, base = packets.get(id(owner), (None, None))
            return _Access(address, False, vector, base)
    return _Access(address, False)


def _names_mentioned(node: ast.AST, renames: Mapping[str, str]) -> set[str]:
    """Canonical names ``node`` mentions in any context (reads AND stores).

    The wrapper-placement caller relies on stores counting (an attached lane
    base definition must hide its coordinate); the expression callers only
    ever see Load-context names.  Not the shared Load-only ``names_read``.
    """
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
    # a name it writes, the loops of every other definition of that name; for
    # a release / acquire atomic's barrier, the atomic's loops.
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
    # The names among ``reads`` the statement may read before it assigns
    # them itself (``_exposed_reads``): a temporary a branch of an ``if``
    # assigns and then reads carries nothing from an earlier run.
    exposed_reads: frozenset[str] = frozenset()
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
    # The tensors whose accesses across the lanes of one lane loop the
    # emitter proved ordered as the program reads them
    # (``HELION_LANE_ORDERED_ATTR``: the staging buffer of an ``hl.split``
    # exchange, checked numerically over the thread block and every lane
    # iteration).  Two statements both ordered on a tensor are not paired on
    # it at the lane level; the thread-axis and device-loop pairs still are.
    lane_ordered: frozenset[str] = frozenset()
    # For an attached statement (``index`` is -1): the indices of the body
    # statements it belongs to, which give it its place in program order.
    sites: tuple[int, ...] = ()

    def positions(self) -> tuple[int, ...]:
        return (self.index,) if self.index >= 0 else self.sites


def _exposed_reads(
    statements: Sequence[ast.AST],
    defined: frozenset[str],
    renames: Mapping[str, str],
    *,
    loop_bodies: bool = False,
) -> frozenset[str]:
    """Canonical names ``statements`` may read before assigning them.

    Only a plain assignment of a whole name defines it, and an ``if`` defines
    what both of its branches do; any other compound statement (a loop) is
    treated as reading everything it reads first, or with ``loop_bodies``
    what its header and its body do (a body reading a name it assigned
    before reads that iteration's value; what it assigns may never run).
    """
    return exposed_reads(
        statements,
        defined=defined,
        renames=renames,
        loops="may_skip" if loop_bodies else "opaque",
    )


def _carried_writes(
    statements: Sequence[ast.AST], renames: Mapping[str, str]
) -> list[frozenset[str]]:
    """Per statement of one loop iteration, the canonical names it assigns
    that the iteration may have read before assigning them.

    The next iteration reads the value such a statement leaves, so repeating
    it once per lane updates a loop-carried variable once per lane: with the
    phi's names merged (``DeviceLoopState.carried_names``), ``c_copy = c``
    ... ``c = c_copy + 1.0``.  A plain copy back of the variable's own value
    from before the iteration (``c = c_copy``) changes nothing and is left
    out, and so is a value a later statement reassigns before anything
    reads it: the default an if-join assigns before an ``if`` whose two
    branches both assign the variable never reaches the next iteration.
    """

    def canonical(name: str) -> str:
        return renames.get(name, name)

    # The names each statement's successors reassign before reading them.
    overwritten: list[frozenset[str]] = []
    dead: frozenset[str] = frozenset()
    for statement in reversed(statements):
        overwritten.append(dead)
        if isinstance(statement, ast.Assign):
            assigned = _assigned_names([statement], renames)
        elif isinstance(statement, ast.If):
            assigned = _assigned_names(statement.body, renames) & _assigned_names(
                statement.orelse, renames
            )
        else:
            assigned = frozenset()
        reads = _exposed_reads([statement], frozenset(), renames, loop_bodies=True)
        dead = (dead - reads) | assigned
    overwritten.reverse()

    result: list[frozenset[str]] = []
    defined: frozenset[str] = frozenset()
    exposed: set[str] = set()
    written: set[str] = set()
    # Locals holding a variable's value from before the iteration.
    entry_values: dict[str, str] = {}
    for statement in statements:
        # ``_exposed_reads`` of the statements so far, one at a time.
        exposed |= _exposed_reads([statement], defined, renames, loop_bodies=True)
        if isinstance(statement, ast.Assign):
            defined |= _assigned_names([statement], renames)
        elif isinstance(statement, ast.If):
            defined |= _assigned_names(statement.body, renames) & _assigned_names(
                statement.orelse, renames
            )
        writes = {canonical(name) for name in ReadWrites.from_ast(statement).writes}
        target = source = None
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and isinstance(statement.value, ast.Name)
        ):
            target = canonical(statement.targets[0].id)
            value = canonical(statement.value.id)
            source = entry_values.get(value, None if value in written else value)
        carried = (writes & exposed) - overwritten[len(result)]
        if target is not None and source == target:
            carried.discard(target)
        result.append(frozenset(carried))
        written |= writes
        for name in writes:
            entry_values.pop(name, None)
        if target is not None and source is not None:
            entry_values[target] = source
    return result


def _assigned_names(
    statements: Sequence[ast.AST], renames: Mapping[str, str]
) -> frozenset[str]:
    """Canonical names a plain top-level assignment in ``statements`` binds."""
    return frozenset(
        renames.get(target.id, target.id)
        for statement in statements
        if isinstance(statement, ast.Assign)
        for target in statement.targets
        if isinstance(target, ast.Name)
    )


def _analyze(index: int, node: ast.AST, renames: Mapping[str, str]) -> _Statement:
    def canonical(names: Iterable[str]) -> frozenset[str]:
        return frozenset(renames.get(name, name) for name in names)

    if getattr(node, HELION_FENCE_BARRIER_ATTR, False):
        # A release / acquire atomic's barrier: a fence touching no tensor and
        # reading no lane, placed in its atomic's lane loops
        # (``_propagate_lanes``) on the atomic's side of every memory access
        # instead of repeating inside every lane loop.
        empty: frozenset[str] = frozenset()
        return _Statement(
            index,
            node,
            empty,
            empty,
            empty,
            empty,
            empty,
            pinned=False,
            register_only=False,
            atomic=False,
            fence=True,
        )

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
        and _is_movable(node, also=_is_atomic_call)
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
        exposed_reads=_exposed_reads([node], frozenset(), renames),
        opaque=canonical(opaque),
        visible=canonical(visible),
        regions={
            renames.get(tensor, tensor): regions
            for tensor, regions in _access_regions(node).items()
        },
        lane_ordered=canonical(getattr(node, HELION_LANE_ORDERED_ATTR, ())),
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
    definition of a name it writes (a release / acquire atomic's barrier
    inside the atomic's loops), and inside the loops that the loops it
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
    fence_partners = _fence_barrier_partners(statements)
    changed = True
    while changed:
        changed = False
        # A release / acquire atomic's barrier runs in the atomic's loops,
        # every thread the same number of times: once per atomic, between
        # the accesses the atomic orders and the atomic itself.
        for barrier, atomic in fence_partners:
            size = len(barrier.lanes)
            barrier.lanes |= atomic.lanes
            changed = changed or len(barrier.lanes) != size
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


def _fence_barrier_partners(
    statements: list[_Statement],
) -> list[tuple[_Statement, _Statement]]:
    """Each release / acquire barrier with the atomic statement it orders.

    The atomic codegen emits a release barrier right before the atomic's
    statements and an acquire barrier right after them
    (``HELION_FENCE_BARRIER_ATTR``), with nothing but its register
    computations in between: the nearest atomic, fence or statement of
    unknown effects after or before the barrier, when that is an atomic.
    """
    partners: list[tuple[_Statement, _Statement]] = []
    for position, statement in enumerate(statements):
        role = getattr(statement.node, HELION_FENCE_BARRIER_ATTR, None)
        if role == "release":
            following = statements[position + 1 :]
        elif role == "acquire":
            following = statements[:position][::-1]
        else:
            continue
        partner = next(
            (
                other
                for other in following
                if other.atomic or other.fence or other.pinned
            ),
            None,
        )
        if partner is not None and partner.atomic:
            partners.append((statement, partner))
    return partners


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


def _conjuncts(test: ast.expr) -> frozenset[str]:
    """The conjuncts of ``test`` (itself, or the operands of its ``and``), unparsed."""
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
        return frozenset().union(*(_conjuncts(value) for value in test.values))
    return frozenset({ast.unparse(test)})
