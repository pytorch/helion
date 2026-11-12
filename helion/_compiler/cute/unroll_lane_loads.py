"""Trace-time lane unrolling with every unrolled lane's loads issued first.

A synthetic lane loop (``DeviceGridState.wrap_body`` for a grid axis, the
device-loop body for a ``hl.tile`` axis inside the kernel, and the lanes of
a looped reduction's chunk or of a persistent reduction row) walks a
thread's elements of one axis one at a time, and the CuTe DSL lowers its
``range(N)`` to an ``scf.for``::

    for lane_0 in range(N):
        indices_0 = tile_offset_0 + cutlass.Int32(lane_0)
        idxs = (indices.iterator + ...).load()
        for lane_1 in range(1):                       # vector partition of another axis
            lane_base_1 = ...
            _tile_unroll_vec_1_0 = cute.arch.load(..., V)
            for vec_lane_1 in cutlass.range_constexpr(V):
                ...compute...
                cute.arch.atomic_add(...)

Lane ``n + 1``'s loads issue after lane ``n``'s compute and memory effects,
so a memory-bound body serializes one DRAM round trip per lane with a single
load in flight per thread.  With ``cute_lane_unroll = U`` this pass unrolls
``U`` consecutive lanes at trace time and issues all of their loads before
the first of them computes::

    _lane_unroll_loads_0_0 = []
    _lane_unroll_loads_0_1 = []
    for lane_0 in cutlass.range_constexpr(N):  # U == N here
        indices_0 = tile_offset_0 + cutlass.Int32(lane_0)
        idxs = (indices.iterator + ...).load()
        _lane_unroll_loads_0_0.append(idxs)
        for lane_1 in cutlass.range_constexpr(1):
            lane_base_1 = ...
            _tile_unroll_vec_1_0 = cute.arch.load(..., V)
            _lane_unroll_loads_0_1.append(_tile_unroll_vec_1_0)
    for lane_0 in cutlass.range_constexpr(N):
        indices_0 = tile_offset_0 + cutlass.Int32(lane_0)
        idxs = _lane_unroll_loads_0_0[lane_0]
        for lane_1 in cutlass.range_constexpr(1):
            lane_base_1 = ...
            _tile_unroll_vec_1_0 = _lane_unroll_loads_0_1[lane_0 * 1 + lane_1]
            for vec_lane_1 in cutlass.range_constexpr(V):
                ...

When ``U < N`` the two constexpr loops sit inside a rolled ``range(N // U)``
loop and the lane index is ``outer * U + inner``.  The trace-time Python
lists only name SSA values; the DSL sees ``U`` independent loads followed by
the compute, which is the order the tile program gives them (a tile load
precedes everything after it for the whole tile, so lane ``n + 1``'s load
moving above lane ``n``'s atomic is the program's order, not a reordering).

A looped reduction's lane computes its vector in a constexpr V-loop
(``HELION_REDUCTION_VLOOP_ATTR``) whose loads before its first memory effect
are tile loads of the chunk, which the program issues before the chunk's
stores: that V-loop is one more level, its element loads moving one copy
per element.

Only loads that precede the body's first memory effect in program order
move (a load after a store or atomic keeps its place, since the tile program
issues it after that effect for every element), together with the pure index
arithmetic they need, which is re-derived in the compute phase.  A V-loop
nested in the body is one more level: its trips are trace-time copies
already, but a scalar load in it (a vector axis that is not the loaded
tensor's contiguous dim, such as a transposed tile's rows) issues after the
previous trip's stores, so its leading loads move into the load phase like a
nested lane loop's.  A nest whose V-loop stores or atomically updates after
such a load takes the split even when its lane loop has a single trip or
cannot be unrolled further (a factor of 1: the lane loop stays rolled around
the two phases).  When the V-loop's copies break the unroll's bounds, the
V-loop stays in place, a statement with memory effects, and the loads ahead
of it still move.  A statement moves only when every name it reads is
defined outside the loop, is a lane variable, or is defined exactly once in
the loop by a statement that moved before it, and when its own name is not
read earlier in the lane body: a loop-carried value or a name defined twice
pins its readers, and a loop-carried name's assignment keeps its place (in
the load phase it would leave the last lane's value where the lane reads
the carried one).
Bodies still holding ``_helion_*`` compiler markers, reversed lane loops and
already-constexpr lane loops are left alone, and so is a body holding any loop
the trace keeps rolled other than a lane loop and a loop of one trip (a looped
reduction's sweep, a serial scan's fallback loop, an inner ``hl.tile`` loop
of several tiles), which every copy would repeat; the copies the knob asks
for trace at most ``_MAX_UNROLLED_STATEMENTS`` statements (``_trace_size``).
A lane loop that a vectorized reduction's cross-lane fold must fully unroll
skips both checks: it is unrolled in place otherwise (``_force_constexpr``).
The pass reports whether it fired so the caller can regenerate with the knob
off when it did not (``LaneUnrollNotApplied`` in ``generate_ast``): the knob
alone never changes the code.
"""

from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING

from ... import exc
from ...autotuner.config_spec import _CUTE_LANE_UNROLL_CHOICES
from ...runtime.config import Config
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from ..ast_read_writes import HELION_VEC_LANE_OF_ATTR
from ..ast_read_writes import ReadWrites
from ._ast_pass_utils import _is_proven_relocatable_call
from ._ast_pass_utils import call_path
from ._ast_pass_utils import loop_extent
from .cache_policy_loads import _CUTE_CACHE_LOAD_HELPER_NAMES
from .fuse_two_pass_loads import _range_bounds
from .fuse_two_pass_loads import _trip_count_for
from .lane_loops.markers import HELION_REDUCTION_VLOOP_ATTR
from .lane_loops.markers import _literal_range_trips
from .lane_loops.owner_guards import _memory_write_calls
from .lane_loops.statements import contains_compiler_marker
from .packet_access import match_packet_access
from .scalar_policy_loads import is_extended_scalar_load
from .scalar_policy_loads import scalar_load_policy

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection
    from collections.abc import Iterable
    from collections.abc import Mapping

_LIST_PREFIX = "_lane_unroll_loads_"


class LaneUnrollNotApplied(exc.Base):
    """``cute_lane_unroll`` named a lane unroll but no lane loop took it.

    The knob must not change the code by itself, so ``generate_ast`` generates
    the kernel again with the knob off (``lane_unroll_off_config``).  An
    ``exc.Base`` so the statement visitor propagates it unwrapped.
    """


def lane_unroll_off_config(config: Config) -> Config:
    """``config`` with the lane unroll off."""
    return Config.from_dict({**config.config, "cute_lane_unroll": 1})


# Trace-time copies of one lane body (unroll factor times the extents of the
# nested lane loops it hoists from) beyond which the pass declines; the
# loads-first split of ``cute/split_lane_loads.py`` shares the bound.
_MAX_UNROLLED_LANES = 64
# Vector packets a thread holds live across the load phase: the unroll
# factor times the packets one lane body loads, each trip of a nested lane
# loop it hoists from included.  Every hoisted packet stays in
# registers until its lane computes: rope's 16-head lane with two 16-byte
# packets per head unrolled 16 ways held 32 packets and cost 206 registers
# per thread (12% of the SM's warps active, 8-10% slower than Triton under a
# cold L2), while 8 ways (16 packets) was the fastest layout measured.
# Larger knob values saturate here instead of spilling; scalar loads (a
# gathered row index) do not count.
_MAX_LIVE_PACKETS = 16
# Statements the trace-time copies of a lane body emit (the copies times the
# ``_trace_size`` of one); the loads-first split shares the bound.  What an
# element costs varies far more than the element count: the split of a
# 15-input BatchNorm-inference body (12 gathered per-channel loads and 3 rsqrt
# per element, 429 statements per 8-element lane) compiled for 7 s at 16
# lanes, 38 s at 32 and 253 s (8 GB) at 64, a 1-input scale at 64 lanes (2752
# statements) for 0.3 s.  The split measures a lane body without its lane base
# and hoisted ``_tile_unroll_vec_*`` loads, which its two phases trace again
# (``split_lane_loads``); the knob measures the whole body, so a body near the
# bound can pass the split's count and fail the knob's.
_MAX_UNROLLED_STATEMENTS = 8192


def _trace_size(
    statements: Iterable[ast.AST], constexpr_values: Mapping[str, int] | None = None
) -> int | None:
    """The statements one trace of ``statements`` emits, each once per trip
    of the ``cutlass.range_constexpr`` loops around it.  A lane loop counts
    its trips too, whether the trace unrolls it or not: one left rolled is
    counted as unrolled, an overcount on the safe side, and its body is the
    per-element work of another tile axis, not a loop with traffic of its own.
    So does a rolled loop of one trip (or none), its static bounds resolved
    through ``constexpr_values``: an inner ``hl.tile`` loop whose block covers
    its extent runs once, as straight-line code would.  A ``range_constexpr``
    whose bound the trace reads (``cute.size(frag)``) counts once, an
    undercount: the trace unrolls it, but its trips are not known here.

    ``None`` when any other loop sits among them: a ``while`` or a ``for`` the
    trace keeps rolled (a looped reduction's sweep, a serial scan's fallback
    loop, an inner ``hl.tile`` loop of several tiles).  Every copy of a lane
    body repeats such a loop as an ``scf.for`` of its own at a cost its
    statements do not show (two sweeps per row over 8-wide chunks with
    register-prefetched reloads compiled for 15 s at 64 copies and 106 s at
    256), while the loads a copy issues first are a few per-lane operands next
    to the loop's own traffic, so neither pass copies it.  Declining a loop
    of a few trips the same way is a compile-time safeguard; its speed was
    not measured.  A loop's trips are counted one way for every kind
    (``_loop_trips``).
    """
    size = 0
    for statement in statements:
        trips = 1
        if isinstance(statement, ast.While):
            return None
        if isinstance(statement, ast.For):
            loop_trips = _loop_trips(statement, constexpr_values)
            if isinstance(statement.iter, ast.Call) and call_path(
                statement.iter.func
            ) == ("cutlass", "range_constexpr"):
                # A bound read at trace time (``cute.size(...)``) counts once.
                trips = 1 if loop_trips is None else loop_trips
            elif _is_lane_loop(statement) and loop_trips is not None:
                trips = loop_trips
            elif not _runs_once(statement, loop_trips):
                return None
        nested = [
            node
            for node in ast.iter_child_nodes(statement)
            if isinstance(node, ast.stmt)
        ]
        inner = _trace_size(nested, constexpr_values)
        if inner is None:
            return None
        size += 1 + trips * inner
    return size


def _loop_trips(
    loop: ast.For, constexpr_values: Mapping[str, int] | None
) -> int | None:
    """The trips of a ``for`` over ``range`` or ``cutlass.range_constexpr``:
    by its literal bounds, any step (``range(N - 1, -1, -1)`` included), else
    by bounds resolved through ``constexpr_values`` (names, through
    ``cutlass.Int32`` casts, and a positive step); ``None`` otherwise."""
    literal = _literal_range_trips(loop)
    if literal is not None:
        return literal
    bounds = _range_bounds(loop.iter)
    if bounds is None:
        return None
    return _trip_count_for(*bounds, dict(constexpr_values or {}))


def _runs_once(loop: ast.For, trips: int | None) -> bool:
    """Whether a ``for`` of ``trips`` trips (``_loop_trips``) runs at most
    once: an ``else`` clause or an unknown count says no."""
    return not loop.orelse and trips is not None and trips <= 1


def _vloop_extent(stmt: ast.AST) -> int | None:
    """The extent of a tile V-loop (a ``cutlass.range_constexpr(V)``,
    ``V >= 2``, that a ``VecLaneWrapper`` tags with the lane loop around it,
    ``HELION_VEC_LANE_OF_ATTR``), else ``None``.  A persistent reduction's
    V-loop is its own lane loop (the tag names its own variable) and is no
    such level, nor is a V-loop an earlier pass rebuilt without the tag (the
    per-lane fold of ``hoist_warp_reduce``)."""
    extent = loop_extent(stmt, kinds=("constexpr",), minimum=2)
    if extent is None:
        return None
    lane = getattr(stmt, HELION_VEC_LANE_OF_ATTR, None)
    target = stmt.target  # type: ignore[attr-defined]
    return extent if lane is not None and lane != target.id else None


def _ascending_range_extent(loop: ast.AST) -> int | None:
    """``for <name> in range(<int>)`` -> the extent; ``None`` for anything else."""
    return loop_extent(loop, kinds=("range",), minimum=1)


def _is_lane_loop(stmt: ast.AST) -> bool:
    return isinstance(stmt, ast.For) and bool(
        getattr(stmt, HELION_LANE_LOOP_VAR_ATTR, None)
    )


def _calls(node: ast.AST) -> list[ast.Call]:
    return [child for child in ast.walk(node) if isinstance(child, ast.Call)]


def _is_load_call(call: ast.Call) -> bool:
    func = call.func
    if isinstance(func, ast.Attribute) and func.attr == "load":
        return True
    name = ast.unparse(func)
    return name == "cute.arch.load" or name in _CUTE_CACHE_LOAD_HELPER_NAMES


def _is_packet_load(stmt: ast.AST) -> bool:
    """Whether the hoisted statement loads a vector packet (not a scalar):
    a plain or L2-hinted ``cute.arch.load`` packet form."""
    return any(
        (access := match_packet_access(call)) is not None
        and access.kind in ("packet", "hinted_packet")
        for call in _calls(stmt)
    )


def _has_extended_scalar_load(loop: ast.For) -> bool:
    """Whether the body loads a scalar through a hinted ``cute.arch.load``
    that reaches NVVM as an extended load (``is_extended_scalar_load``).

    The CuTe DSL's compiler aborts (a native abort, no diagnostic) on the
    trace-time unrolled form of a body holding such loads: the SDK's
    load/store vectorizer groups the extended loads with ordinary ones.  The
    unroll declines such a body and keeps the knob-off code.  The 4- and
    8-byte scalar hints are no such loads: ``lower_scalar_policy_loads``
    spells them as exact-policy PTX, which the vectorizer leaves alone.
    """
    return any(is_extended_scalar_load(call) for call in _calls(loop))


def _has_lowered_scalar_load(loop: ast.For) -> bool:
    """Whether the body holds a hinted 4- or 8-byte scalar load, one that
    ``lower_scalar_policy_loads`` spells as exact-policy PTX
    (``scalar_load_policy``)."""
    return any(scalar_load_policy(call) is not None for call in _calls(loop))


def _is_effect_free(node: ast.AST) -> bool:
    """Whether every call in ``node`` is a proven pure computation or a load."""
    return all(
        _is_proven_relocatable_call(call, allow_load=True) for call in _calls(node)
    )


def _candidate_target(stmt: ast.AST) -> str | None:
    """The name a plain single-target assignment binds, when its right-hand
    side is register arithmetic and loads only."""
    if (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and _is_effect_free(stmt.value)
    ):
        return stmt.targets[0].id
    return None


def _nested_level(stmt: ast.AST, vloops: bool) -> tuple[int, bool] | None:
    """``(extent, tile V-loop)`` of a statement the planner takes as one
    more level, else ``None``: a nested lane loop; a tile V-loop
    (``_vloop_extent``, when ``vloops``), whose scalar loads issue after the
    previous trip's stores; or a looped reduction's V-loop
    (``HELION_REDUCTION_VLOOP_ATTR``), whose loads before its first memory
    effect are tile loads of the chunk too.  A V-loop's loads move one copy
    per element."""
    if getattr(stmt, HELION_REDUCTION_VLOOP_ATTR, False):
        extent = loop_extent(stmt, kinds=("constexpr",))
        return None if extent is None else (extent, False)
    if vloops and (extent := _vloop_extent(stmt)) is not None:
        return extent, True
    if _is_lane_loop(stmt) and (extent := _ascending_range_extent(stmt)) is not None:
        return extent, False
    return None


@dataclasses.dataclass
class _Hoisted:
    """One statement of a lane body that moves into the load phase."""

    stmt: ast.Assign
    name: str
    # Index of the trace-time list forwarding this value into the compute
    # phase; ``None`` for pure index arithmetic, which is re-derived there.
    list_index: int | None


@dataclasses.dataclass
class _LevelPlan:
    """The load-phase content of one lane loop level."""

    loop: ast.For
    extent: int
    # Original statements in order, with hoisted ones and nested levels.
    items: list[_Hoisted | _LevelPlan]
    # A V-loop level (``_vloop_extent``) rather than a lane loop.
    vloop: bool = False

    def load_count(self) -> int:
        return sum(
            (1 if item.list_index is not None else 0)
            if isinstance(item, _Hoisted)
            else item.load_count()
            for item in self.items
        )

    def packet_count(self) -> int:
        """Hoisted vector loads (``cute.arch.load(..., VectorType)`` and the
        cache-hinted packet helpers) of one trip of this level: a nested
        level's once per trip of it, as the load phase appends every one."""
        return sum(
            (1 if item.list_index is not None and _is_packet_load(item.stmt) else 0)
            if isinstance(item, _Hoisted)
            else item.extent * item.packet_count()
            for item in self.items
        )

    def vloop_split(self) -> bool:
        """Whether a nested V-loop level hoists a load that a memory write of
        that V-loop follows: the split then moves the load ahead of the
        write, which pays off without any copies of the lane body.  A V-loop
        that only collects values (a packet's flush comes after it) already
        issues its loads back to back."""
        return any(
            isinstance(item, _LevelPlan)
            and ((item.vloop and item.writes_after_load()) or item.vloop_split())
            for item in self.items
        )

    def writes_after_load(self) -> bool:
        """Whether a store or atomic of this level's loop body follows one of
        the loads the level hoists."""
        hoisted = {
            id(item.stmt)
            for item in self.items
            if isinstance(item, _Hoisted) and item.list_index is not None
        }
        load_seen = False
        for stmt in self.loop.body:
            if load_seen and _memory_write_calls(stmt):
                return True
            load_seen = load_seen or id(stmt) in hoisted
        return False

    def nested_product(self) -> int:
        """Trace-time copies of the nested levels of one trip of this level:
        a nested level's trips times its own copies, added over sibling
        levels, which the load phase runs one after another; 1 without any."""
        return (
            sum(
                item.extent * item.nested_product()
                for item in self.items
                if isinstance(item, _LevelPlan)
            )
            or 1
        )


class _Planner:
    """Decide which statements of one lane loop nest move into the load phase."""

    def __init__(self, loop: ast.For, *, vloops: bool = True) -> None:
        # Whether tile V-loops are nested levels (``_vloop_extent``).
        self.vloops = vloops
        rw = ReadWrites.from_ast(loop)
        # Names written anywhere in the nest (``visit_For`` skips loop
        # targets, so lane variables are not counted here).
        self.write_counts: dict[str, int] = dict(rw.writes)
        self.next_list = 0
        # For each statement of the nest, the names it reads or an earlier
        # statement of its body reads, or one ahead of its loop in an
        # enclosing body: a lane reads them before it assigns them.
        self.read_before: dict[int, set[str]] = {}
        self._record_reads(loop.body, set())

    def _record_reads(self, body: list[ast.stmt], before: set[str]) -> None:
        read = before
        for stmt in body:
            self.read_before[id(stmt)] = {*read, *ReadWrites.from_ast(stmt).reads}
            if isinstance(stmt, ast.For):
                self._record_reads(stmt.body, read)
            read = self.read_before[id(stmt)]

    def plan(
        self,
        loop: ast.For,
        extent: int,
        available: set[str],
        outer_pending: list[dict[str, ast.Assign]],
    ) -> tuple[_LevelPlan, list[str]]:
        """Plan one level.

        ``available``: names the load phase has defined when this level runs
        (outer lane variables and outer statements already hoisted).
        ``outer_pending``: the enclosing levels' pure candidates that could
        still move, innermost last.  Returns the plan and the names of outer
        pending candidates this level's hoists read, in first-read order, so
        the caller hoists them deterministically before this level's load
        loop.
        """
        lane_var = loop.target.id  # type: ignore[union-attr]
        available = {*available, lane_var}
        items: list[_Hoisted | _LevelPlan] = []
        effect_seen = False
        # Pure candidates of this level seen so far that could move if a
        # load needs them, in order; a load pulls its backward slice.
        pending: dict[str, ast.Assign] = {}
        hoisted: set[str] = set()
        # Ordered: the caller hoists these in this order, which must not
        # depend on string hashing.
        outer_deps: dict[str, None] = {}

        def reachable(name: str) -> bool:
            return (
                self.write_counts.get(name, 0) == 0
                or name in available
                or name in hoisted
                or name in pending
                or any(name in level for level in outer_pending)
            )

        def hoistable(stmt: ast.AST, target: str) -> bool:
            # A name the lane reads before it assigns it (``acc = acc + 1``,
            # or ``w = idx * 2`` ahead of ``idx = ...``) carries the value the
            # previous lane left: assigned in the load phase, it would hold
            # the last lane's value when the compute phase reads it.
            if (
                self.write_counts.get(target, 0) != 1
                or target in self.read_before[id(stmt)]
            ):
                return False
            return all(reachable(name) for name in ReadWrites.from_ast(stmt).reads)

        def pull(names: Iterable[str]) -> None:
            """Move the pending pure dependencies (transitively) first."""
            for name in names:
                if name in hoisted or name in available:
                    continue
                dependency = pending.pop(name, None)
                if dependency is not None:
                    pull(ReadWrites.from_ast(dependency).reads)
                    items.append(_Hoisted(dependency, name, None))
                    hoisted.add(name)
                elif any(name in level for level in outer_pending):
                    outer_deps[name] = None

        for stmt in loop.body:
            level = None if effect_seen else _nested_level(stmt, self.vloops)
            if level is not None:
                nested_extent, is_vloop = level
                nested, nested_deps = self.plan(
                    stmt,  # type: ignore[arg-type]
                    nested_extent,
                    {*available, *hoisted},
                    [*outer_pending, pending],
                )
                nested.vloop = is_vloop
                if nested.load_count():
                    pull(nested_deps)
                    items.append(nested)
                if not _is_effect_free(stmt):
                    effect_seen = True
                continue
            target = _candidate_target(stmt)
            if target is not None:
                assert isinstance(stmt, ast.Assign)
                has_load = any(_is_load_call(call) for call in _calls(stmt.value))
                if not has_load:
                    if hoistable(stmt, target):
                        pending[target] = stmt
                    continue
                if effect_seen or not hoistable(stmt, target):
                    continue
                pull(ReadWrites.from_ast(stmt).reads)
                items.append(_Hoisted(stmt, target, self.next_list))
                self.next_list += 1
                hoisted.add(target)
                continue
            if not _is_effect_free(stmt):
                effect_seen = True
        return _LevelPlan(loop, extent, items), list(outer_deps)


def _index_text(index_vars: list[tuple[str, int]]) -> str:
    """Trace-time list index of the current lane copy from ``(var, extent)``
    pairs, outermost first: ``(outer * K1 + inner1) * K2 + inner2``."""
    text = ""
    for var, extent in index_vars:
        text = var if not text else f"({text}) * {extent} + {var}"
    return text


def _clone_setup(stmt: ast.AST) -> ast.stmt:
    """Duplicate a pure statement (ExtendedAST nodes cannot be deep-copied)."""
    return statement_from_string(ast.unparse(stmt))


def _emit_load_phase(
    plan: _LevelPlan,
    lists: list[str],
    index_vars: list[tuple[str, int]],
    lane_var: str | None,
) -> ast.For:
    """The load-phase loop of one level: hoisted statements in order, loads
    bound under their own names and appended to their lists."""
    loop_var = lane_var or plan.loop.target.id  # type: ignore[union-attr]
    body: list[ast.stmt] = []
    for item in plan.items:
        if isinstance(item, _Hoisted):
            body.append(_clone_setup(item.stmt))
            if item.list_index is not None:
                body.append(
                    statement_from_string(
                        f"{lists[item.list_index]}.append({item.name})"
                    )
                )
        else:
            nested_var = item.loop.target.id  # type: ignore[union-attr]
            body.append(
                _emit_load_phase(
                    item, lists, [*index_vars, (nested_var, item.extent)], None
                )
            )
    loop = statement_from_string(
        f"for {loop_var} in cutlass.range_constexpr({plan.extent}):\n    pass"
    )
    assert isinstance(loop, ast.For)
    loop.body = body
    return loop


def _rewrite_compute_phase(
    plan: _LevelPlan, lists: list[str], index_vars: list[tuple[str, int]]
) -> None:
    """Rewrite the original loop in place: constexpr iterator, hoisted loads
    read back from their lists (pure setup stays as it was)."""
    replacements: dict[int, str] = {}
    nested: dict[int, _LevelPlan] = {}
    for item in plan.items:
        if isinstance(item, _Hoisted):
            if item.list_index is not None:
                replacements[id(item.stmt)] = (
                    f"{item.name} = {lists[item.list_index]}[{_index_text(index_vars)}]"
                )
        else:
            nested[id(item.loop)] = item
    new_body: list[ast.stmt] = []
    for stmt in plan.loop.body:
        text = replacements.get(id(stmt))
        if text is not None:
            new_body.append(statement_from_string(text))
            continue
        level = nested.get(id(stmt))
        if level is not None:
            nested_var = level.loop.target.id  # type: ignore[union-attr]
            _rewrite_compute_phase(
                level, lists, [*index_vars, (nested_var, level.extent)]
            )
        new_body.append(stmt)  # type: ignore[arg-type]
    plan.loop.body = new_body
    iterator = expr_from_string(f"cutlass.range_constexpr({plan.extent})")
    assert isinstance(iterator, ast.expr)
    plan.loop.iter = iterator


def _unroll_factor(
    extent: int,
    unroll: int,
    nested_product: int,
    packet_count: int,
    statements: int,
) -> int:
    """The largest unroll choice within ``unroll`` and the bounds, 0 when
    not even a single lane body fits them."""
    factor = 0
    for choice in _CUTE_LANE_UNROLL_CHOICES:
        if (
            choice <= unroll
            and extent % choice == 0
            and choice * nested_product <= _MAX_UNROLLED_LANES
            and choice * packet_count <= _MAX_LIVE_PACKETS
            and choice * statements <= _MAX_UNROLLED_STATEMENTS
        ):
            factor = choice
    return factor


def _plan_unroll(
    loop: ast.For,
    extent: int,
    unroll: int,
    statements: int,
    *,
    require_full: bool,
    vloops: bool,
) -> tuple[_Planner, _LevelPlan, int] | None:
    """A plan of ``loop``'s loads-first unroll and its factor, or ``None``
    when it hoists no load or no factor within ``unroll`` keeps its copies,
    packets and traced statements within the bounds.  A forced unroll needs
    every lane (``require_full``).  A factor of 1 (the lane loop stays rolled
    around the two phases) applies only when a V-loop's loads split from
    the writes that follow them (``_LevelPlan.vloop_split``), which needs
    V-loop levels (``vloops``), never planned for a forced unroll."""
    planner = _Planner(loop, vloops=vloops)
    plan, _deps = planner.plan(loop, extent, set(), [])
    smallest = 1 if plan.vloop_split() else 2
    factor = _unroll_factor(
        extent, unroll, plan.nested_product(), plan.packet_count(), statements
    )
    if (
        not plan.load_count()
        or factor < smallest
        or (require_full and factor != extent)
    ):
        return None
    return planner, plan, factor


def _unroll_one(
    loop: ast.For,
    unroll: int,
    new_var: Callable[[str], str],
    *,
    require_full: bool = False,
    constexpr_values: Mapping[str, int] | None = None,
) -> list[ast.stmt] | None:
    extent = _ascending_range_extent(loop)
    if (
        extent is None
        or contains_compiler_marker(list(loop.body))
        or _has_extended_scalar_load(loop)
        # A forced unroll (the cross-lane fold's) runs whatever the knob
        # says, so it copies a body holding a hinted scalar load in place:
        # moving those loads first would change the code without the knob.
        or (require_full and _has_lowered_scalar_load(loop))
    ):
        return None
    # A forced unroll copies the body in place otherwise (``_force_constexpr``).
    statements = 0 if require_full else _trace_size(loop.body, constexpr_values)
    if statements is None:
        return None
    # V-loops are nested levels unless their copies break the bounds; the
    # plan without them, which keeps a V-loop in place, stands then.  A
    # forced unroll (the cross-lane fold's) runs whatever the knob says, so
    # it keeps its V-loops in place: moving their loads first would change
    # the code without the knob.
    planned = None
    if not require_full:
        planned = _plan_unroll(
            loop, extent, unroll, statements, require_full=False, vloops=True
        )
    planned = planned or _plan_unroll(
        loop, extent, unroll, statements, require_full=require_full, vloops=False
    )
    if planned is None:
        return None
    planner, plan, factor = planned
    lists = [new_var(f"{_LIST_PREFIX}{i}") for i in range(planner.next_list)]
    lane_var = loop.target.id  # type: ignore[union-attr]
    list_inits = [statement_from_string(f"{name} = []") for name in lists]
    if factor == extent:
        load_loop = _emit_load_phase(plan, lists, [(lane_var, extent)], None)
        _rewrite_compute_phase(plan, lists, [(lane_var, extent)])
        _drop_lane_attr(loop)
        return [*list_inits, load_loop, loop]
    # Partial unroll: ``factor`` lanes per trip of a rolled outer loop; the
    # lane variable becomes ``outer * factor + inner`` in both phases.
    outer_var = new_var(f"{lane_var}_blk")
    inner_var = new_var(f"{lane_var}_u")
    plan.extent = factor
    load_loop = _emit_load_phase(plan, lists, [(inner_var, factor)], inner_var)
    load_loop.body.insert(
        0, statement_from_string(f"{lane_var} = {outer_var} * {factor} + {inner_var}")
    )
    _rewrite_compute_phase(plan, lists, [(inner_var, factor)])
    loop.target = ast.Name(id=inner_var, ctx=ast.Store())
    loop.body.insert(
        0, statement_from_string(f"{lane_var} = {outer_var} * {factor} + {inner_var}")
    )
    ast.fix_missing_locations(loop)
    outer = statement_from_string(
        f"for {outer_var} in range({extent // factor}):\n    pass"
    )
    assert isinstance(outer, ast.For)
    outer.body = [*list_inits, load_loop, loop]
    _drop_lane_attr(loop)
    return [outer]


def _drop_lane_attr(loop: ast.For) -> None:
    """The rewritten nest is no longer a canonical lane loop (its trips are
    trace-time copies); later lane-loop passes must not pick it up."""
    for node in ast.walk(loop):
        if isinstance(node, ast.For) and hasattr(node, HELION_LANE_LOOP_VAR_ATTR):
            delattr(node, HELION_LANE_LOOP_VAR_ATTR)


def _force_constexpr(loop: ast.For) -> list[ast.stmt]:
    """Unroll a lane loop at trace time in place, loads where they are.

    The fallback of a forced unroll (``forced_full``) when the loads-first
    form declines: a body without hoistable loads, or more packets than
    ``_MAX_LIVE_PACKETS`` (each fold then re-reads the packets its slice
    needs rather than holding the whole slice live across the folds).  A
    body holding compiler markers is rejected: the passes expanding them
    match the rolled nest.  So is an extent beyond ``_MAX_UNROLLED_LANES``:
    the trace-time copies of the lane body (and ptxas's time on them) grow
    with the lane count, and the loads-first form is bounded the same way.
    """
    extent = _ascending_range_extent(loop)
    if extent is None or contains_compiler_marker(list(loop.body)):
        raise exc.BackendUnsupported(
            "cute",
            f"lane loop {loop.target.id} over a vectorized reduction axis "  # type: ignore[union-attr]
            "cannot be unrolled for its cross-lane fold",
        )
    if extent > _MAX_UNROLLED_LANES:
        raise exc.BackendUnsupported(
            "cute",
            f"lane loop {loop.target.id} walks {extent} packets per thread; "  # type: ignore[union-attr]
            f"the cross-lane fold unrolls at most {_MAX_UNROLLED_LANES}",
        )
    iterator = expr_from_string(f"cutlass.range_constexpr({extent})")
    assert isinstance(iterator, ast.expr)
    loop.iter = iterator
    _drop_lane_attr(loop)
    return [loop]


def unroll_lane_loads(
    body: list[ast.stmt],
    unroll: int,
    new_var: Callable[[str], str],
    *,
    forced_full: Collection[str] = (),
    constexpr_values: Mapping[str, int] | None = None,
) -> tuple[list[ast.stmt], bool]:
    """Apply the loads-first lane unroll to every lane loop reachable from
    ``body``; returns the new body and whether the knob transformed any loop.
    ``constexpr_values`` resolves the bounds of rolled loops (``_trace_size``).

    ``forced_full`` names lane variables whose loops are unrolled completely
    whatever the knob says (the vectorized lane loops of a reduced block,
    whose per-lane reductions ``cute/fold_lane_reductions.py`` folds across
    trace-time lanes): loads first when that form applies, in place
    otherwise.  Those unrolls do not count as the knob firing.
    """
    fired = False
    forced = frozenset(forced_full)

    def visit(statements: list[ast.stmt], knob: int) -> list[ast.stmt]:
        nonlocal fired
        result: list[ast.stmt] = []
        for stmt in statements:
            if _is_lane_loop(stmt):
                lane_var = getattr(stmt, HELION_LANE_LOOP_VAR_ATTR)
                extent = _ascending_range_extent(stmt)
                if lane_var in forced and extent is not None and extent >= 2:
                    replacement = _unroll_one(
                        stmt,  # type: ignore[arg-type]
                        extent,
                        new_var,
                        require_full=True,
                    )
                    if replacement is None:
                        replacement = _force_constexpr(stmt)  # type: ignore[arg-type]
                    result.extend(replacement)
                    continue
                if knob > 1:
                    replacement = _unroll_one(
                        stmt,  # type: ignore[arg-type]
                        knob,
                        new_var,
                        constexpr_values=constexpr_values,
                    )
                    if replacement is not None:
                        fired = True
                        result.extend(replacement)
                        continue
                # A lane loop kept rolled: the knob unrolls a nest from its
                # outermost loop only, but a forced loop nested in it (the
                # vectorized reduction axis under a row lane loop) still
                # unrolls.
                if forced:
                    stmt.body = visit(stmt.body, 1)  # type: ignore[assignment]
                    stmt.orelse = visit(stmt.orelse, 1)  # type: ignore[assignment]
            elif isinstance(stmt, (ast.For, ast.If, ast.While)):
                stmt.body = visit(stmt.body, knob)  # type: ignore[assignment]
                stmt.orelse = visit(stmt.orelse, knob)  # type: ignore[assignment]
            result.append(stmt)
        return result

    if unroll <= 1 and not forced:
        return body, False
    return visit(body, unroll), fired
