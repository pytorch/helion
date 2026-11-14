"""The shared memory a rendered CuTe kernel allocates, and the per-block check.

Role: analysis.

The DSL gives every ``cute.arch.alloc_smem`` it traces a static allocation of
its own, laid out side by side in the kernel's shared memory: a call in a body
the DSL unrolls at trace time (a ``cutlass.range_constexpr`` loop, a loop over
a Python sequence) allocates once per iteration, one in a dynamic loop
(``range``, ``cutlass.range``, ``while``) once, one in a ``cutlass.const_expr``
branch only when the branch is taken, and one in either branch of any other
``if``.  The device helpers a body calls allocate the same way: every traced
call of a shared-memory grouped reduce (``reduce_helpers``) takes a buffer of
its own, so the 16 unrolled copies of a vector lane loop hold 16 of them.
The compiler then drops the allocations of code it proves never runs: a
branch whose test folds (to a Python value, a DSL constant, or by range
analysis: ``tile_offset_0 < 0`` with the offset built from the block index), a
loop whose range or test is empty on entry (the constant bounds of an empty
dimension, a bound times zero), and an allocation nothing reads.  A kernel
whose remaining allocations add up past the per-block maximum fails in libNVVM
("NVVM backend compilation failed"): the opt-in maximum from sm_90 (227 KiB on
sm_90 and sm_100, 99 KiB on sm_120), 48 KiB of static allocations below
(``target_facts.max_smem_per_block``).

``smem_footprint`` bounds what a rendered body allocates.  ``low`` adds the
allocations it can size that certainly survive: the grouped reduces of such
code (one whose buffer goes with an unread result, the column reduce's, only
where such code reads the result: ``_UNREAD_KEPT_SCOPES``), and the literal
``alloc_smem`` calls bound to a name whose pointer such code reads, directly
or through views of it (``cute.make_tensor``, ``cute.recast_ptr``, an offset,
an ``iterator``: a view nothing reads uses no storage), of code no test the
compiler might fold guards, each times the trip counts of the unrolled loops
around it.  Such code is the kernel's straight-line body, a dynamic loop whose
range has a trip for some values of its bounds that the enclosing code allows
(static bounds, or the kernel's runtime scalar arguments, which the enclosing
loops may already bound: ``range(r, 0)`` never runs inside ``range(0, r)``),
the persistent loop over a static, nonempty set of virtual program ids (block
0 runs its first), the lesser branch of an ``if`` (a read counts when both
branches make it), and every branch of the root dispatch of a multi-root
launch whose roots all get blocks (``RootDispatch``); a ``while`` body, a loop
over other runtime bounds or under a runtime test, the code after a statement
that may ``return``, and the greater branch count in ``high`` only, as does an
allocation bound any other way (an argument of another call, a list element,
an annotated assignment's value).  A read counts for the binding it sees (the
walk visits each body last statement first): not for one a later statement
replaces, nor for a view the branches of a foldable test bind to other
storage, and an unrolled loop's allocation counts only where each trip reads
its own copy.  A store through a subscript of a name or a view of it
(``t[i] = v``, ``t.iterator[i] = v``) or through a view's ``store`` method
reads nothing, since the compiler drops the stores to storage no kept code
loads, nor does a read of a view's size or layout (``cute.size``, ``.shape``),
nor an assignment to names nothing reads, which it drops with its loads.  A
load the compiler folds counts all the same: one of storage nothing stores to,
or whose value folds away (``x * 0``); generated code writes neither.
``high`` is ``None`` as soon as an allocation it cannot size may be traced (a
size it cannot evaluate, storage the DSL's ``SmemAllocator`` or the dynamic
base reserves, an unrolled trip count it cannot read).  The
``cute_pdl_spread`` pad is sized against these bounds (``passes.pdl``,
``pdl_release.spread_pad_bytes``), and ``check_smem_budget`` rejects a kernel
whose ``low`` exceeds the target's per-block maximum before the DSL compiles
it, so a ``low`` past what the compiler keeps would refuse a kernel that
compiles.  Of the device helpers only the grouped reduces are modelled (with
``_cute_resident_sums``, whose storage is opaque): the helpers the generic
emitters call on every config.  ``padding`` bounds the alignment padding ahead
of the allocations ``high`` counts (each allocation's ``alignment`` less one,
its element's width by default), which the spread pad counts against the
per-block maximum and in the density it targets; ``low`` counts none, staying
a lower bound.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import functools
import itertools
import math
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import cast

from ... import exc
from ..compile_environment import CompileEnvironment
from ..program_id import ForEachProgramID
from ..program_id import L2GroupingProgramIDs
from ..program_id import PersistentProgramIDs
from . import target_facts
from ._ast_pass_utils import call_path
from ._ast_pass_utils import callee_name
from ._ast_pass_utils import is_staged_loop
from ._ast_pass_utils import is_trace_time_test
from .reduce_call import ReduceScope
from .reduce_call import shared_columns_elements
from .reduce_call import use_interleaved_shared_reduce
from .reduce_call import use_serial_block_reduce

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet
    from typing import Literal

    from ..program_id import ProgramIDs

# Element widths of the ``cutlass`` numeric types a body allocates shared
# memory in (``cute.arch.alloc_smem(cutlass.<type>, count, ...)``) and of the
# identities that type a grouped reduce's buffer.
_CUTLASS_TYPE_BYTES: dict[str, int] = {
    "cutlass.Int8": 1,
    "cutlass.Uint8": 1,
    "cutlass.Int16": 2,
    "cutlass.Uint16": 2,
    "cutlass.Float16": 2,
    "cutlass.BFloat16": 2,
    "cutlass.Int32": 4,
    "cutlass.Uint32": 4,
    "cutlass.Float32": 4,
    "cutlass.TFloat32": 4,
    "cutlass.Int64": 8,
    "cutlass.Uint64": 8,
    "cutlass.Float64": 8,
}

# Calls through which a body reserves shared memory of a size this walk
# cannot read: the DSL's ``SmemAllocator`` (its ``allocate`` /
# ``allocate_tensor`` / ``allocate_array`` carve a struct or a layout's bytes
# out of the kernel's allocation at trace time; the fa4 flash families and
# the tcgen05 GEMMs hold their rings this way, at the per-block maximum
# already) and the dynamic shared memory base.
_OPAQUE_SMEM_CALLS = frozenset(
    {
        "SmemAllocator",
        "allocate",
        "allocate_tensor",
        "allocate_array",
        "get_dyn_smem",
    }
)

# The integer casts generated code wraps sizes and loop bounds in.
_INTEGER_CASTS = frozenset({("cutlass", "Int32"), ("cutlass", "Int64")})

# The keywords of ``cutlass.range`` that leave its trip count alone (unroll
# and pipelining hints, an at-least-once annotation).
_TRIP_NEUTRAL_KEYWORDS = frozenset(
    {
        "unroll",
        "unroll_full",
        "prefetch_stages",
        "pipelining",
        "vectorize",
        "at_least_once",
    }
)

# List methods that change a list's length (a name bound to a list literal
# keeps the literal's length only while none of them is called on it).
_RESIZING_METHODS = frozenset({"append", "extend", "insert", "pop", "remove", "clear"})

# The grouped-reduce dispatchers by the name generated code calls them by.
_REDUCE_SCOPES: dict[str, ReduceScope] = {scope.value: scope for scope in ReduceScope}

# Calls whose result views the storage of their first argument: binding one
# reads nothing, and the storage is used only once the view is.
_VIEW_CALLS = frozenset({"make_tensor", "recast_ptr", "recast_tensor"})

# Reads of a view's layout or type, which touch none of its storage: the
# calls by their first argument, the attributes by the object.  Offline for
# sm_100a, a buffer whose only read is ``cute.size`` of a view compiles past
# the per-block maximum.
_METADATA_CALLS = frozenset(
    {("cute", "size"), ("cute", "cosize"), ("cute", "rank"), ("cute", "depth")}
)
_METADATA_ATTRIBUTES = frozenset(
    {"shape", "stride", "layout", "element_type", "memspace"}
)


# Nodes whose body may be traced any number of times: definitions (once per
# call) and comprehensions (once per element).
_REPEATED_NODES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


@dataclasses.dataclass(frozen=True)
class SmemFootprint:
    """Bounds of the shared memory (bytes) a rendered body allocates: at
    least ``low``, at most ``high`` (``None`` when no bound is known), and at
    most ``padding`` bytes of alignment padding ahead of the allocations
    ``high`` counts (each allocation's alignment less one)."""

    low: int = 0
    high: int | None = 0
    padding: int = 0

    def __add__(self, other: SmemFootprint) -> SmemFootprint:
        high = (
            None if self.high is None or other.high is None else self.high + other.high
        )
        return SmemFootprint(self.low + other.low, high, self.padding + other.padding)

    def repeated(self, fewest: int, most: int | None) -> SmemFootprint:
        """This footprint traced at least ``fewest`` and at most ``most``
        times (``None``: a count the walk cannot read)."""
        if most is None:
            return SmemFootprint(self.low * fewest, 0 if self.high == 0 else None)
        high = None if self.high is None else self.high * most
        return SmemFootprint(self.low * fewest, high, self.padding * most)

    def uncertain(self) -> SmemFootprint:
        """This footprint where none of it is certain: in ``high`` only."""
        return dataclasses.replace(self, low=0)


_UNKNOWN = SmemFootprint(0, None)


@dataclasses.dataclass(frozen=True)
class RootDispatch:
    """The ``if pid < threshold`` tests a multi-root launch dispatches its
    roots by, when every root gets blocks: ``pid`` names the shared program
    id and ``thresholds`` holds the leading roots' block counts, accumulated.
    Each branch of such a test runs on some block, so the kernel keeps
    both."""

    pid: str
    thresholds: frozenset[int]


@dataclasses.dataclass(frozen=True)
class _Facts:
    """What the compiler may know about the kernel's runtime scalar arguments
    in the code the walk is in: the interval of each scalar an enclosing loop
    bounds against a static bound (entering ``range(0, r)`` proves ``r >=
    1``), the scalars an enclosing loop bounds against each other
    (``related``: the walk keeps no relations), and whether a runtime test
    guards the code (``guarded``: it may tell the compiler anything)."""

    intervals: Mapping[str, tuple[float, float]] = MappingProxyType({})
    related: frozenset[str] = frozenset()
    guarded: bool = False

    def interval(self, bound: int | str) -> tuple[float, float]:
        """The values a range bound may take: a static one its own, a
        runtime scalar those of its interval."""
        if isinstance(bound, int):
            return (bound, bound)
        return self.intervals.get(bound, (-math.inf, math.inf))


def _is_alloc_smem(call: ast.Call) -> bool:
    """Whether ``call`` is a ``cute.arch.alloc_smem``."""
    return isinstance(call.func, ast.Attribute) and call.func.attr == "alloc_smem"


def _view_source(node: ast.expr) -> str | ast.Call | None:
    """What ``node`` views: the name whose storage it views (the name itself,
    ``_VIEW_CALLS`` of it, an offset from it, its ``iterator``), the
    ``alloc_smem`` call whose pointer it views, or None when ``node`` is not
    a view."""
    root = _view_root(node)
    return root.id if isinstance(root, ast.Name) else root


def _view_root(node: ast.expr) -> ast.Name | ast.Call | None:
    """``_view_source`` of ``node``, the name as the node that names it."""
    while True:
        if isinstance(node, ast.Name):
            return node
        if isinstance(node, ast.Call):
            if _is_alloc_smem(node):
                return node
            if callee_name(node.func) not in _VIEW_CALLS or not node.args:
                return None
            node = node.args[0]
        elif isinstance(node, ast.Attribute) and node.attr == "iterator":
            node = node.value
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            node = node.left
        else:
            return None


def _assigned_names(node: ast.AST) -> list[str] | None:
    """The names the assignment ``node`` binds when it binds names only
    (``x = ...``, ``x, *y = ...``, ``x += ...``); None for any other
    statement, a store through a subscript or an attribute among them."""
    pending: list[ast.expr]
    if isinstance(node, ast.Assign):
        pending = list(node.targets)
    elif isinstance(node, ast.AugAssign):
        pending = [node.target]
    else:
        return None
    names: list[str] = []
    while pending:
        target = pending.pop()
        if isinstance(target, ast.Starred):
            target = target.value
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            pending.extend(target.elts)
        else:
            return None
    return names


def _stored_names(node: ast.AST) -> set[str]:
    """The names the statement ``node`` stores through and reads nowhere
    else: the assignment ``t[i] = v`` through the name or a view of it
    (``t.iterator[i] = v``; ``t[i] += v`` loads ``t``), and the store method
    of a view of the name (``(p + i).store(v)``)."""
    stored: set[str] = set()
    others: list[ast.AST] = []
    if isinstance(node, ast.Assign):
        others.append(node.value)
        for target in node.targets:
            root = (
                _view_root(target.value) if isinstance(target, ast.Subscript) else None
            )
            if isinstance(target, ast.Subscript) and isinstance(root, ast.Name):
                stored.add(root.id)
                others.append(target.slice)
                # The offsets and layouts of the view, its root aside.
                others.extend(
                    name
                    for name in ast.walk(target.value)
                    if isinstance(name, ast.Name) and name is not root
                )
            else:
                others.append(target)
    elif (
        isinstance(node, ast.Expr)
        and isinstance(call := node.value, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "store"
        and isinstance(root := _view_root(call.func.value), ast.Name)
    ):
        stored.add(root.id)
        others.extend(call.args)
        others.extend(keyword.value for keyword in call.keywords)
        # The offsets and layouts of the view, its root aside.
        others.extend(
            name
            for name in ast.walk(call.func.value)
            if isinstance(name, ast.Name) and name is not root
        )
    for other in others:
        for name in ast.walk(other):
            if isinstance(name, ast.Name):
                stored.discard(name.id)
    return stored


# The grouped reduces whose buffer survives though nothing reads their result,
# its stores ordered by their barriers: offline for sm_100a, 32 unread copies
# of the fragment two-stage reduce (270336 bytes), 64 of the tree reduce
# (262400) and of the two-stage reduce's interleaved block form (278528), and
# 32 of its generic form (307200), each fail in libNVVM.  An unread column
# reduce's buffer goes (64 fp32 values over 1024 threads, 262144 bytes,
# compile).
_UNREAD_KEPT_SCOPES = frozenset(
    {
        ReduceScope.SHARED_TWO_STAGE,
        ReduceScope.SHARED_TREE,
        ReduceScope.SHARED_TWO_STAGE_FRAGMENT,
    }
)


def _keeps_buffer_unread(node: ast.AST) -> bool:
    """Whether the statement ``node`` assigns or evaluates a call of a grouped
    reduce in ``_UNREAD_KEPT_SCOPES``."""
    value = node.value if isinstance(node, (ast.Assign, ast.Expr)) else None
    return (
        isinstance(value, ast.Call)
        and _REDUCE_SCOPES.get(callee_name(value.func) or "") in _UNREAD_KEPT_SCOPES
    )


def _may_return(node: ast.AST) -> bool:
    """Whether the statement ``node`` holds a ``return`` from the body it is
    in (one of a function it defines aside)."""
    pending = [node]
    while pending:
        current = pending.pop()
        if isinstance(current, ast.Return):
            return True
        if not isinstance(
            current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            pending.extend(ast.iter_child_nodes(current))
    return False


def _strip_integer_casts(node: ast.expr) -> ast.expr:
    """``node`` without the ``cutlass.Int32`` / ``cutlass.Int64`` casts
    around it."""
    while (
        isinstance(node, ast.Call)
        and call_path(node.func) in _INTEGER_CASTS
        and len(node.args) == 1
        and not node.keywords
    ):
        node = node.args[0]
    return node


def _buffer(
    element_type: str | None, elements: int | None, alignment: int | None = None
) -> SmemFootprint:
    """``elements`` values of the ``cutlass`` type named ``element_type``,
    aligned to ``alignment`` bytes (by default, as ``cute.arch.alloc_smem``
    aligns, to the element's width)."""
    width = _CUTLASS_TYPE_BYTES.get(element_type or "")
    if width is None or elements is None:
        return _UNKNOWN
    return SmemFootprint(width * elements, width * elements, (alignment or width) - 1)


def _branches(
    body: SmemFootprint,
    orelse: SmemFootprint,
    traced: Literal["one", "both", "either"],
) -> SmemFootprint:
    """The two branches of an ``if``: the kernel keeps the allocations of
    ``"one"`` (a ``cutlass.const_expr`` test traces only it), of ``"both"``
    (the root dispatch, ``RootDispatch``), and at least those of the lesser
    for ``"either"`` (any other test: both are traced, and the compiler drops
    the branch a test it folds never takes)."""
    if traced == "both":
        return body + orelse
    low = min(body.low, orelse.low)
    if body.high is None or orelse.high is None:
        return SmemFootprint(low, None)
    if traced == "one":
        return SmemFootprint(
            low, max(body.high, orelse.high), max(body.padding, orelse.padding)
        )
    return SmemFootprint(low, body.high + orelse.high, body.padding + orelse.padding)


def _literal_length(node: ast.expr) -> int | None:
    """The length of a list or tuple literal without starred elements."""
    if isinstance(node, (ast.List, ast.Tuple)) and not any(
        isinstance(element, ast.Starred) for element in node.elts
    ):
        return len(node.elts)
    return None


def _sequence_lengths(statements: Iterable[ast.AST]) -> dict[str, int]:
    """The names bound exactly once, by an assignment of a list or tuple
    literal, whose length nothing changes afterwards (no resizing method
    call, slice assignment or ``del`` of an element), by that length: the
    per-element accumulator lists a register tile hands its shared-memory
    column reduce."""
    bindings: dict[str, int] = {}
    lengths: dict[str, int] = {}
    resized: set[str] = set()
    for statement in statements:
        for node in ast.walk(statement):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                bindings[node.id] = bindings.get(node.id, 0) + 1
            elif (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                length = _literal_length(node.value)
                if length is not None:
                    lengths[node.targets[0].id] = length
            elif isinstance(node, ast.Subscript) and (
                isinstance(node.ctx, ast.Del)
                or (
                    isinstance(node.ctx, ast.Store)
                    and isinstance(node.slice, ast.Slice)
                )
            ):
                if isinstance(node.value, ast.Name):
                    resized.add(node.value.id)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _RESIZING_METHODS
                and isinstance(node.func.value, ast.Name)
            ):
                resized.add(node.func.value.id)
    return {
        name: length
        for name, length in lengths.items()
        if bindings.get(name) == 1 and name not in resized
    }


def _shared_two_stage_elements(pre: int, group_span: int, group_count: int) -> int:
    """The elements of the buffer ``_cute_grouped_reduce_shared_two_stage``
    allocates, by the form it routes the group shape to (the predicates it
    shares with ``reduce_call``): the serial form one slot per warp, the
    interleaved block form (from 32 siblings) every thread's value plus the
    results, the interleaved warp form and the two-stage form the per-warp
    partials of every sibling plus the results."""
    warps = group_span // 32
    if use_serial_block_reduce(pre, group_span, group_count):
        return warps
    if use_interleaved_shared_reduce(pre, group_span) and pre >= 32:
        return group_count * (group_span + pre)
    return group_count * pre * (warps + 1)


class _Walk:
    """One ``smem_footprint`` walk over a rendered body."""

    def __init__(
        self,
        statements: list[ast.AST],
        constexpr_values: Mapping[str, int],
        runtime_scalars: AbstractSet[str],
        dispatch: RootDispatch | None,
        persistent_loop: str | None,
    ) -> None:
        self.constexpr_values = constexpr_values
        self.runtime_scalars = runtime_scalars
        self.dispatch = dispatch
        self.persistent_loop = persistent_loop
        self.statements = statements
        self.facts = _Facts()
        # The names whose current binding the code after the node being
        # walked certainly reads (``nodes`` walks a body last statement
        # first): an allocation bound to one of them survives.
        self.live: set[str] = set()
        # Whether the innermost loop being walked has a ``break`` or
        # ``continue`` (the ``else`` clause of a loop in it still belongs to
        # it).
        self.exits = False
        # ``bound`` by node id, filled as the walk visits statements.
        self._bound: dict[int, frozenset[str]] = {}
        # The ids of the ``alloc_smem`` calls an assignment binds to names,
        # directly or through views (``_view_source``): any other allocation
        # counts in ``high`` only.
        self._named_allocations = {
            id(source)
            for statement in statements
            for node in ast.walk(statement)
            if isinstance(node, ast.Assign)
            and all(isinstance(target, ast.Name) for target in node.targets)
            and isinstance(source := _view_source(node.value), ast.Call)
        }

    @functools.cached_property
    def sequence_lengths(self) -> dict[str, int]:
        """``_sequence_lengths`` of the body, read only by a loop over a name
        or a column reduce of one."""
        return _sequence_lengths(self.statements)

    @contextlib.contextmanager
    def within(self, facts: _Facts) -> Iterator[None]:
        """Walk the code under ``facts`` (a loop's body, an ``if``'s
        branches), then restore the enclosing code's."""
        enclosing, self.facts = self.facts, facts
        try:
            yield
        finally:
            self.facts = enclosing

    def nodes(self, nodes: Iterable[ast.AST]) -> SmemFootprint:
        """The footprint of the statements ``nodes``, walked last first: on
        entry ``live`` holds the names the code after them certainly reads,
        on exit those the code from their first statement on does."""
        total = SmemFootprint()
        for node in reversed(list(nodes)):
            if _may_return(node):
                # The code after a statement that may return is not certainly
                # traced, nor its reads certainly made.
                total = total.uncertain()
                self.live = set()
            total += self.node(node)
        return total

    def loop_entry(self, loop: ast.For) -> _Facts | None:
        """The facts inside the dynamic ``loop`` when it certainly has a trip
        for some block (``entry``): the persistent loop over virtual program
        ids always does (block 0 starts at the first of a static, nonempty
        set); None when the walk cannot show a trip."""
        target = loop.target
        if (
            self.persistent_loop is not None
            and isinstance(target, ast.Name)
            and target.id == self.persistent_loop
        ):
            return self.facts
        return self.entry(loop.iter)

    def node(self, node: ast.AST) -> SmemFootprint:
        if isinstance(node, (ast.For, ast.While)):
            return self.loop(node)
        if isinstance(node, ast.If):
            return self.conditional(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            footprint = self.repeated(node)
            self.live.discard(node.name)
            return footprint
        if any(
            isinstance(child, (ast.stmt, ast.excepthandler, ast.match_case))
            for child in ast.iter_child_nodes(node)
        ):
            # A compound statement the walk does not model counts in ``high``
            # only, and the reads in it stay out of ``live``, so ``low`` stays
            # a lower bound: a ``try`` or ``match`` (generated code has none),
            # and a ``with``, which generated code writes for the
            # ``cute.arch.elect_one()`` regions of the tcgen05 and flash
            # kernels.  The compiler keeps such a region, so ``low`` misses an
            # allocation only the region reads, and a spread pad that tops
            # ``low`` up takes more L1 carve-out than it needs.
            enclosing, self.live = self.live, set()
            try:
                footprint = self.children(node)
            finally:
                self.live = enclosing - self.bound(node)
            return footprint.uncertain()
        return self.statement(node)

    def loop(self, node: ast.For | ast.While) -> SmemFootprint:
        # The ``else`` clause runs after the last trip.
        orelse = self.nodes(node.orelse)
        after = self.live
        inside = self.facts
        if isinstance(node, ast.While):
            # A staged ``while`` traces its body once, which the compiler
            # drops when it proves the test false on entry.
            head = node.test
            trips = (0, None) if is_trace_time_test(node.test) else (0, 1)
        elif is_staged_loop(node):
            # The DSL traces a dynamic loop's body once, and the compiler
            # keeps it unless it proves the range empty.
            head, entered = node.iter, self.loop_entry(node)
            trips = (0, 1) if entered is None else (1, 1)
            inside = entered or inside
        else:
            head, trips = node.iter, self.trips(node)
        # A single traced copy of the body (a dynamic loop's, or one unrolled
        # trip) reads what the code after the loop reads.  Several unrolled
        # copies each keep their own allocations, and the code after the
        # loop reads only the last copy's bindings: a copy must read its
        # allocations itself.
        self.live = set(after) if trips == (1, 1) else set()
        enclosing_exits, self.exits = self.exits, False
        with self.within(inside):
            body = self.nodes(node.body)
        exits_early, self.exits = self.exits, enclosing_exits
        certain = trips[0] >= 1 and not exits_early
        loop = body.repeated(*trips) + orelse
        if exits_early:
            # An unrolled loop keeps Python's ``break`` and ``continue`` (the
            # DSL rejects them in a dynamic one): the rest of an iteration,
            # the later ones and the ``else`` may never be traced, so the
            # trips bound the allocations from above only.
            loop = loop.uncertain()
        targets = self.bound(node.target) if isinstance(node, ast.For) else frozenset()
        replaced = targets.union(*map(self.bound, node.body))
        self._bound[id(node)] = replaced.union(*map(self.bound, node.orelse))
        if trips[1] == 0:
            live = after
        elif certain and trips[1] == 1:
            live = self.live - targets
        else:
            # The code after the loop reads, of the bindings from before it,
            # those no trip may replace, and a body that certainly runs
            # reads what its first trip reads.
            live = (after - replaced) | (self.live - targets if certain else set())
        head_footprint, reads, _ = self.expression(head)
        self.live = live | reads
        return head_footprint + loop

    def conditional(self, node: ast.If) -> SmemFootprint:
        traced = self.traced(node.test)
        guarded = dataclasses.replace(self.facts, guarded=True)
        after = self.live
        with self.within(guarded if traced == "either" else self.facts):
            self.live = set(after)
            body = self.nodes(node.body)
            body_live, self.live = self.live, set(after)
            orelse = self.nodes(node.orelse)
        # A binding stays certainly read when both branches leave it read,
        # or either does under the root dispatch, which keeps them all.
        if traced == "both":
            self.live |= body_live
        else:
            self.live &= body_live
        test_footprint, reads, binds = self.expression(node.test)
        self.live |= reads
        self._bound[id(node)] = frozenset(binds).union(
            *map(self.bound, node.body), *map(self.bound, node.orelse)
        )
        return test_footprint + _branches(body, orelse, traced)

    def statement(self, node: ast.AST) -> SmemFootprint:
        """A plain statement: its footprint, given the names the code after
        it certainly reads (``live``), which it updates to the names the code
        from it on certainly reads."""
        footprint, reads, binds = self.expression(node)
        live = self.live
        if isinstance(node, (ast.Assign, ast.Expr)):
            source = _view_source(node.value)
            if isinstance(source, ast.Call):
                # The DSL drops an allocation, bound directly or through
                # views, whose pointer nothing reads (the spread pad reads
                # its own through a keep helper).
                names = (
                    [
                        target.id
                        for target in node.targets
                        if isinstance(target, ast.Name)
                    ]
                    if isinstance(node, ast.Assign)
                    else []
                )
                if live.isdisjoint(names):
                    footprint = footprint.uncertain()
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            reads.add(node.target.id)
        elif isinstance(node, (ast.Break, ast.Continue, ast.Return)):
            self.exits = True
        assigned = _assigned_names(node)
        unread = isinstance(node, ast.Expr) or (
            assigned is not None and live.isdisjoint(assigned)
        )
        if unread and not _keeps_buffer_unread(node):
            # A call whose result nothing reads may lose its storage with it:
            # offline for sm_100a, an unread column reduce's buffer goes.
            footprint = footprint.uncertain()
        if assigned is not None and live.isdisjoint(assigned):
            # The compiler drops an assignment to names nothing reads, with
            # the loads in it: binding a view reads nothing of the storage it
            # views, nor does a load whose value is never used.
            reads.clear()
        else:
            # A store keeps nothing: the compiler drops the stores to, and
            # then the allocation of, storage no kept code loads.
            reads -= _stored_names(node)
        self._bound[id(node)] = frozenset(binds)
        live -= binds
        live |= reads
        return footprint

    def expression(self, node: ast.AST) -> tuple[SmemFootprint, set[str], set[str]]:
        """The footprint of the plain expression or statement ``node`` (its
        calls, and the conditional expressions and repeated forms in it by
        their rules), the names it certainly reads (those both branches of a
        conditional expression read, either under the root dispatch, and none
        in a repeated form) and the names it binds."""
        footprint = SmemFootprint()
        reads: set[str] = set()
        binds: set[str] = set()
        # The roots of the views whose metadata alone the code reads
        # (``_METADATA_CALLS``, ``_METADATA_ATTRIBUTES``): no storage.
        metadata_roots: set[int] = set()
        # An explicit stack: a long operator chain must not recurse.
        pending = [node]
        while pending:
            current = pending.pop()
            view = (
                current.args[0]
                if isinstance(current, ast.Call)
                and call_path(current.func) in _METADATA_CALLS
                and current.args
                else current.value
                if isinstance(current, ast.Attribute)
                and current.attr in _METADATA_ATTRIBUTES
                else None
            )
            if view is not None and isinstance(root := _view_root(view), ast.Name):
                metadata_roots.add(id(root))
            if isinstance(current, ast.IfExp):
                traced = self.traced(current.test)
                test, test_reads, test_binds = self.expression(current.test)
                body, body_reads, body_binds = self.expression(current.body)
                orelse, orelse_reads, orelse_binds = self.expression(current.orelse)
                footprint += test + _branches(body, orelse, traced)
                reads |= test_reads
                if traced == "both":
                    reads |= body_reads | orelse_reads
                else:
                    reads |= body_reads & orelse_reads
                binds |= test_binds | body_binds | orelse_binds
                continue
            if isinstance(current, _REPEATED_NODES):
                footprint += self.repeated(current)
                continue
            if isinstance(current, ast.Call):
                footprint += self.call(current)
            elif isinstance(current, ast.Name):
                if not isinstance(current.ctx, ast.Load):
                    binds.add(current.id)
                elif id(current) not in metadata_roots:
                    reads.add(current.id)
            pending.extend(ast.iter_child_nodes(current))
        return footprint, reads, binds

    def repeated(self, node: ast.AST) -> SmemFootprint:
        """A definition allocates whenever it is called, a comprehension once
        per element: counts the walk cannot read.  It may never run, so
        nothing in it is certainly read."""
        enclosing, self.live = self.live, set()
        try:
            return self.children(node).repeated(0, None)
        finally:
            self.live = enclosing

    def children(self, node: ast.AST) -> SmemFootprint:
        """The footprint of what ``node`` holds: its statements, expressions
        and the clauses around them."""
        footprint = SmemFootprint()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.stmt):
                footprint += self.node(child)
            elif isinstance(child, ast.expr):
                footprint += self.expression(child)[0]
            else:
                footprint += self.children(child)
        return footprint

    def bound(self, node: ast.AST) -> frozenset[str]:
        """The names ``node`` may bind in the code around it (not those local
        to a definition or comprehension in it)."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return frozenset({node.name})
        cached = self._bound.get(id(node))
        if cached is None:
            names: set[str] = set()
            pending = [node]
            while pending:
                current = pending.pop()
                if current is not node and isinstance(current, ast.stmt):
                    names |= self.bound(current)
                elif isinstance(current, ast.Name):
                    if not isinstance(current.ctx, ast.Load):
                        names.add(current.id)
                elif not isinstance(current, _REPEATED_NODES):
                    pending.extend(ast.iter_child_nodes(current))
            cached = self._bound[id(node)] = frozenset(names)
        return cached

    def traced(self, test: ast.expr) -> Literal["one", "both", "either"]:
        """Which branches of a conditional on ``test`` the kernel keeps
        (``_branches``)."""
        if is_trace_time_test(test):
            return "one"
        if self.dispatches(test):
            return "both"
        return "either"

    def dispatches(self, test: ast.expr) -> bool:
        """Whether ``test`` is one of the root dispatch's ``pid <
        threshold`` tests."""
        dispatch = self.dispatch
        return (
            dispatch is not None
            and isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == dispatch.pid
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Lt)
            and self.integer(test.comparators[0]) in dispatch.thresholds
        )

    def trips(self, loop: ast.For) -> tuple[int, int | None]:
        """The copies of the body of the unrolled ``loop`` the DSL traces,
        fewest and most (``None``: a count the walk cannot read): one per
        iteration of ``range_constexpr`` and of Python iteration."""
        iterable = loop.iter
        if isinstance(iterable, ast.Call):
            count = (
                self.range_trips(iterable)
                if callee_name(iterable.func) == "range_constexpr"
                else None
            )
        elif isinstance(iterable, ast.Name):
            count = self.sequence_lengths.get(iterable.id)
        else:
            count = _literal_length(iterable)
        return (0, None) if count is None else (count, count)

    def entry(self, iterable: ast.expr) -> _Facts | None:
        """The facts inside a dynamic loop over ``iterable`` that has a trip
        for some values of its bounds the enclosing code allows, so the
        compiler cannot drop it: static bounds with a trip, or bounds among
        which runtime scalar arguments (``runtime_scalars``, each read once,
        neither related by an enclosing loop nor under a runtime test) can
        make one, with what entering the loop proves about them.  None when
        the walk cannot show such values."""
        facts = self.facts
        bounds = self.range_bounds(iterable)
        if bounds is None:
            return None
        if all(isinstance(bound, int) for bound in bounds):
            return facts if self.range_trips(iterable) else None
        names = [bound for bound in bounds if isinstance(bound, str)]
        if (
            facts.guarded
            or len(set(names)) < len(names)
            or not facts.related.isdisjoint(names)
        ):
            return None
        start, stop = (0, bounds[0]) if len(bounds) == 1 else bounds[:2]
        step = bounds[2] if len(bounds) == 3 else 1
        if not isinstance(step, int) or step == 0:
            return None
        if step < 0:
            # The loop runs while the index exceeds its stop.
            start, stop = stop, start
        if facts.interval(start)[0] >= facts.interval(stop)[1]:
            return None
        # Inside, ``start < stop``.
        if isinstance(start, str) and isinstance(stop, str):
            return dataclasses.replace(facts, related=facts.related | {start, stop})
        if isinstance(stop, str):
            assert isinstance(start, int)
            low, high = facts.interval(stop)
            narrowed = {stop: (max(low, start + 1), high)}
        else:
            assert isinstance(start, str)
            low, high = facts.interval(start)
            narrowed = {start: (low, min(high, stop - 1))}
        return dataclasses.replace(facts, intervals={**facts.intervals, **narrowed})

    def range_bounds(self, iterable: ast.expr) -> list[int | str] | None:
        """The one to three bounds of a ``range``-style call (keywords that
        leave its trip count alone aside): static integers (``integer``) or
        the runtime scalar arguments they name through integer casts; None
        otherwise."""
        if (
            not isinstance(iterable, ast.Call)
            or not 1 <= len(iterable.args) <= 3
            or any(
                keyword.arg not in _TRIP_NEUTRAL_KEYWORDS
                for keyword in iterable.keywords
            )
        ):
            return None
        bounds: list[int | str] = []
        for argument in iterable.args:
            bound = self.integer(argument)
            if bound is None:
                bound = self.runtime_scalar(argument)
            if bound is None:
                return None
            bounds.append(bound)
        return bounds

    def range_trips(self, iterable: ast.expr) -> int | None:
        """The length of the range a ``range``-style call spells when
        ``integer`` reads all its bounds; None otherwise."""
        bounds = self.range_bounds(iterable)
        if bounds is None or bounds[2:] == [0]:
            return None
        static = [bound for bound in bounds if isinstance(bound, int)]
        if len(static) < len(bounds):
            return None
        return len(range(*static))

    def runtime_scalar(self, node: ast.expr) -> str | None:
        """The runtime scalar argument ``node`` names, through integer
        casts."""
        node = _strip_integer_casts(node)
        if isinstance(node, ast.Name) and node.id in self.runtime_scalars:
            return node.id
        return None

    def integer(self, node: ast.expr) -> int | None:
        """The static value of an integer literal, a constexpr name, or
        arithmetic over them, through ``cutlass.Int32`` / ``cutlass.Int64``
        casts; None otherwise."""
        node = _strip_integer_casts(node)
        if isinstance(node, ast.Constant):
            return node.value if type(node.value) is int else None
        if isinstance(node, ast.Name):
            return self.constexpr_values.get(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            operand = self.integer(node.operand)
            return None if operand is None else -operand
        if isinstance(node, ast.BinOp):
            left, right = self.integer(node.left), self.integer(node.right)
            if left is None or right is None:
                return None
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, (ast.FloorDiv, ast.Mod)) and right != 0:
                return (
                    left // right if isinstance(node.op, ast.FloorDiv) else left % right
                )
        return None

    def call(self, call: ast.Call) -> SmemFootprint:
        """What one traced ``call`` allocates itself (its arguments aside)."""
        name = callee_name(call.func)
        if name in _OPAQUE_SMEM_CALLS:
            return _UNKNOWN
        if _is_alloc_smem(call):
            if len(call.args) < 2:
                return _UNKNOWN
            alignment_node = (
                call.args[2]
                if len(call.args) > 2
                else next(
                    (item.value for item in call.keywords if item.arg == "alignment"),
                    None,
                )
            )
            alignment = None
            if alignment_node is not None:
                alignment = self.integer(alignment_node)
                if alignment is None:
                    return _UNKNOWN
            buffer = _buffer(
                ast.unparse(call.args[0]), self.integer(call.args[1]), alignment
            )
            if id(call) not in self._named_allocations:
                return buffer.uncertain()
            return buffer
        if name == "_cute_resident_sums":
            # Above one warp its partials take a ``SmemAllocator`` tensor
            # sized by the register fragment it reduces.
            threads = self.integer(call.args[1]) if len(call.args) > 1 else None
            if threads is not None and threads <= 32:
                return SmemFootprint()
            return _UNKNOWN
        scope = _REDUCE_SCOPES.get(name) if name is not None else None
        if scope is not None:
            return self.reduce_buffer(call, scope)
        return SmemFootprint()

    def reduce_buffer(self, call: ast.Call, scope: ReduceScope) -> SmemFootprint:
        """The buffer one traced call of a grouped-reduce dispatcher allocates
        (typed like its identity), sized as the dispatcher sizes it; the warp
        and cluster forms allocate none (a cluster exchange receives into
        buffers the body allocates)."""
        if scope in (
            ReduceScope.WARP,
            ReduceScope.CLUSTER,
            ReduceScope.CLUSTER_WARP_ONLINE_PAIR,
        ):
            return SmemFootprint()
        keywords = {item.arg: item.value for item in call.keywords}

        def keyword(name: str, default: int | None = None) -> int | None:
            value = keywords.get(name)
            return default if value is None else self.integer(value)

        def identity_type(index: int) -> str | None:
            identity = call.args[index] if len(call.args) > index else None
            if not isinstance(identity, ast.Call):
                return None
            return ast.unparse(identity.func)

        pre = keyword("pre", 1)
        group_span = keyword("group_span")
        group_count = keyword("group_count", 1)
        if pre is None or group_span is None or group_count is None:
            return _UNKNOWN
        if scope is ReduceScope.CLUSTER_ONLINE_PAIR:
            # The per-thread sum is block-reduced in an fp32 two-stage first.
            return _buffer(
                "cutlass.Float32", _shared_two_stage_elements(1, group_span, 1)
            )
        if scope in (ReduceScope.SHARED_TWO_STAGE, ReduceScope.BLOCK):
            element_type = identity_type(2)
            elements = _shared_two_stage_elements(pre, group_span, group_count)
        elif scope is ReduceScope.SHARED_TREE:
            element_type = identity_type(2)
            num_threads = keyword("num_threads")
            if num_threads is None:
                return _UNKNOWN
            elements = num_threads + group_count * pre
        elif scope is ReduceScope.SHARED_COLUMNS:
            element_type = identity_type(2)
            values = call.args[0] if call.args else None
            if isinstance(values, ast.Name):
                width = self.sequence_lengths.get(values.id)
            else:
                width = _literal_length(values) if values is not None else None
            fold_value = keywords.get("fold")
            fold = (
                "serial"
                if fold_value is None
                else fold_value.value
                if isinstance(fold_value, ast.Constant)
                else None
            )
            if width is None or fold not in ("serial", "warp", "cooperative"):
                return _UNKNOWN
            elements = shared_columns_elements(
                pre, width, group_span, group_count, fold
            )
        else:
            assert scope is ReduceScope.SHARED_TWO_STAGE_FRAGMENT
            element_type = identity_type(3)
            count = keyword("count")
            if count is None:
                return _UNKNOWN
            elements = group_count * pre * count * (group_span // 32 + 1)
        return _buffer(element_type, elements)


def smem_footprint(
    statements: Iterable[ast.AST],
    constexpr_values: Mapping[str, int] | None = None,
    *,
    runtime_scalars: AbstractSet[str] = frozenset(),
    dispatch: RootDispatch | None = None,
    persistent_loop: str | None = None,
) -> SmemFootprint:
    """Bounds of the shared memory the rendered ``statements`` allocate
    (the module docstring); ``constexpr_values`` resolves constexpr names in
    sizes and trip counts, ``runtime_scalars`` names the kernel arguments
    the compiler cannot know (a dynamic loop bounded by them may run),
    ``dispatch`` the launch's root dispatch (see ``root_dispatch()``) and
    ``persistent_loop`` the variable of a persistent loop every launch
    enters (see ``persistent_loop()``)."""
    nodes = list(statements)
    walk = _Walk(
        nodes, constexpr_values or {}, runtime_scalars, dispatch, persistent_loop
    )
    return walk.nodes(nodes)


def root_dispatch(pid: ProgramIDs | None) -> RootDispatch | None:
    """The root dispatch of a multi-root launch (``ForEachProgramID``) whose
    roots all get blocks; None for any other launch.  The compiler may drop
    the branch of a root with no blocks (an empty first root's ``if
    pid_shared < 0``, a later one's division by a zero block count), so a
    block count that is zero or not static declines, as does a persistent
    launch: its dispatch compares virtual program ids, counted per root,
    while a persistent case's ``static_grid_size`` is the persistent grid."""
    if (
        not isinstance(pid, ForEachProgramID)
        or not pid.cases
        or pid.cases[0]._is_persistent()
    ):
        return None
    counts: list[int] = []
    for case in pid.cases:
        count = case.static_grid_size()
        if count is None or count < 1:
            return None
        counts.append(count)
    thresholds = frozenset(itertools.accumulate(counts[:-1]))
    return RootDispatch(pid.shared_pid_var, thresholds)


def persistent_loop(pid: ProgramIDs | None) -> str | None:
    """The variable of the loop over virtual program ids of a persistent
    launch (the strategy under an L2 grouping, or that of a multi-root
    launch's first root, which loops over every root's ids) whose ids are a
    static, nonempty set; None for any other launch.  Block 0 starts at the
    first id in both the interleaved and the blocked form, so the compiler
    keeps the loop's body, whatever its runtime bounds."""
    cases = pid.cases if isinstance(pid, ForEachProgramID) else [pid]
    strategies: list[PersistentProgramIDs] = []
    for case in cases:
        if isinstance(case, L2GroupingProgramIDs):
            case = case.parent_strategy
        if not isinstance(case, PersistentProgramIDs):
            return None
        strategies.append(case)
    if not strategies:
        return None
    total = 0
    for strategy in strategies:
        counts = [info.static_num_pids() for info in strategy.pid_info]
        if None in counts:
            return None
        total += math.prod(cast("list[int]", counts))
    if total < 1:
        return None
    return strategies[0].virtual_pid_var


def _runtime_scalars(definition: ast.FunctionDef) -> frozenset[str]:
    """The arguments of the kernel ``definition`` its launch passes as
    runtime values that its body never rebinds: the unannotated ones (a
    generated kernel annotates only its ``Constexpr`` arguments), the sizes
    of a kernel without static shapes among them."""
    rebound = {
        node.id
        for node in ast.walk(definition)
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load)
    }
    return frozenset(
        argument.arg
        for argument in definition.args.args
        if argument.annotation is None and argument.arg not in rebound
    )


def kernel_smem_footprint(
    definition: ast.FunctionDef,
    constexpr_values: Mapping[str, int],
    pid: ProgramIDs | None = None,
) -> SmemFootprint:
    """``smem_footprint`` of the kernel ``definition``'s body, given its
    runtime scalar arguments (``_runtime_scalars``) and the root dispatch
    and persistent loop of the launch's program ids ``pid``."""
    return smem_footprint(
        definition.body,
        constexpr_values,
        runtime_scalars=_runtime_scalars(definition),
        dispatch=root_dispatch(pid),
        persistent_loop=persistent_loop(pid),
    )


def check_smem_budget(
    definition: ast.FunctionDef,
    constexpr_values: Mapping[str, int],
    pid: ProgramIDs | None = None,
) -> None:
    """Reject the kernel ``definition`` when the shared memory it certainly
    allocates (``kernel_smem_footprint(...).low``) exceeds what one block
    of the target may hold, which libNVVM would otherwise fail on: an
    unrolled vector lane loop around a shared-memory reduce allocates one
    buffer per copy."""
    allocated = kernel_smem_footprint(definition, constexpr_values, pid).low
    env = CompileEnvironment.current()
    capacity = target_facts.max_smem_per_block(
        env.device, env.config_spec.target_device_capability
    )
    if allocated > capacity:
        raise exc.BackendUnsupported(
            "cute",
            f"a kernel allocating {allocated} bytes of shared memory, over the "
            f"{capacity}-byte per-block maximum (every traced allocation, each "
            "unrolled copy of a shared-memory reduce included, takes its own "
            "buffer)",
        )
