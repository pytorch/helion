"""Trace-time lane unrolling with every unrolled lane's loads issued first.

A synthetic grid lane loop (``DeviceGridState.wrap_body``) walks a thread's
elements of one tile axis one at a time, and the CuTe DSL lowers its
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

Only loads that precede the body's first memory effect in program order
move (a load after a store or atomic keeps its place, since the tile program
issues it after that effect for every element), together with the pure index
arithmetic they need, which is re-derived in the compute phase.  A statement
moves only when every name it reads is defined outside the loop, is a lane
variable, or is defined exactly once in the loop by a statement that moved
before it: a loop-carried value or a name defined twice pins its readers.
Bodies still holding ``_helion_*`` compiler markers, reversed lane loops and
already-constexpr lane loops are left alone.  The pass reports whether it
fired so the caller can regenerate with the knob off when it did not
(``LaneUnrollNotApplied`` in ``generate_ast``): the knob alone never changes
the code.
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
from ..ast_read_writes import ReadWrites
from ..tile_strategy import _is_proven_relocatable_call
from .cache_policy_loads import _CUTE_CACHE_LOAD_HELPER_NAMES
from .lane_loop_distribution import contains_compiler_marker

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable

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
# nested lane loops it hoists from) beyond which the pass declines.
_MAX_UNROLLED_LANES = 64


def _ascending_range_extent(loop: ast.For) -> int | None:
    """``for <name> in range(<int>)`` -> the extent; ``None`` for anything else."""
    iterator = loop.iter
    if (
        not isinstance(loop.target, ast.Name)
        or loop.orelse
        or not isinstance(iterator, ast.Call)
        or not isinstance(iterator.func, ast.Name)
        or iterator.func.id != "range"
        or iterator.keywords
        or len(iterator.args) != 1
    ):
        return None
    arg = iterator.args[0]
    if not isinstance(arg, ast.Constant) or not isinstance(arg.value, int):
        return None
    return arg.value if arg.value >= 1 else None


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


def _has_scalar_policy_load(loop: ast.For) -> bool:
    """Whether the body loads a scalar through ``cute.arch.load`` with a
    cache-policy keyword (``cop=`` / ``level1_eviction_priority=``).

    The CuTe DSL's compiler aborts (a native abort, no diagnostic) on the
    trace-time unrolled form of such a body (repro:
    ``/tmp/segred/run_gen.py /tmp/segred/gen_crash1.py asis`` on 2026-09-29,
    cutlass 4.7); the rolled form and the 16-byte vector helper forms
    compile, so the unroll declines these bodies and the knob-off code is
    kept.
    """
    for call in _calls(loop):
        if not call.keywords or ast.unparse(call.func) != "cute.arch.load":
            continue
        if len(call.args) >= 2 and (
            isinstance(call.args[1], ast.Call)
            and ast.unparse(call.args[1].func) == "ir.VectorType.get"
        ):
            continue
        return True
    return False


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

    def hoisted_names(self) -> set[str]:
        names: set[str] = set()
        for item in self.items:
            if isinstance(item, _Hoisted):
                names.add(item.name)
            else:
                names |= item.hoisted_names()
        return names

    def load_count(self) -> int:
        return sum(
            (1 if item.list_index is not None else 0)
            if isinstance(item, _Hoisted)
            else item.load_count()
            for item in self.items
        )

    def nested_product(self) -> int:
        product = 1
        for item in self.items:
            if isinstance(item, _LevelPlan):
                product *= item.extent * item.nested_product()
        return product


class _Planner:
    """Decide which statements of one lane loop nest move into the load phase."""

    def __init__(self, loop: ast.For) -> None:
        rw = ReadWrites.from_ast(loop)
        # Names written anywhere in the nest (``visit_For`` skips loop
        # targets, so lane variables are not counted here).
        self.write_counts: dict[str, int] = dict(rw.writes)
        self.next_list = 0

    def plan(
        self,
        loop: ast.For,
        extent: int,
        available: set[str],
        outer_pending: list[dict[str, ast.Assign]],
    ) -> tuple[_LevelPlan, set[str]]:
        """Plan one level.

        ``available``: names the load phase has defined when this level runs
        (outer lane variables and outer statements already hoisted).
        ``outer_pending``: the enclosing levels' pure candidates that could
        still move, innermost last.  Returns the plan and the names of outer
        pending candidates this level's hoists read, which the caller must
        hoist before this level's load loop.
        """
        lane_var = loop.target.id  # type: ignore[union-attr]
        available = {*available, lane_var}
        items: list[_Hoisted | _LevelPlan] = []
        effect_seen = False
        # Pure candidates of this level seen so far that could move if a
        # load needs them, in order; a load pulls its backward slice.
        pending: dict[str, ast.Assign] = {}
        hoisted: set[str] = set()
        outer_deps: set[str] = set()

        def reachable(name: str) -> bool:
            return (
                self.write_counts.get(name, 0) == 0
                or name in available
                or name in hoisted
                or name in pending
                or any(name in level for level in outer_pending)
            )

        def hoistable(stmt: ast.AST, target: str) -> bool:
            if self.write_counts.get(target, 0) != 1:
                return False
            return all(
                reachable(name)
                for name in ReadWrites.from_ast(stmt).reads
                if name != target
            )

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
                    outer_deps.add(name)

        for stmt in loop.body:
            if _is_lane_loop(stmt) and not effect_seen:
                nested_extent = _ascending_range_extent(stmt)  # type: ignore[arg-type]
                if nested_extent is not None:
                    nested, nested_deps = self.plan(
                        stmt,  # type: ignore[arg-type]
                        nested_extent,
                        {*available, *hoisted},
                        [*outer_pending, pending],
                    )
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
        return _LevelPlan(loop, extent, items), outer_deps


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


def _unroll_factor(extent: int, unroll: int, nested_product: int) -> int:
    factor = 1
    for choice in _CUTE_LANE_UNROLL_CHOICES:
        if (
            choice <= unroll
            and extent % choice == 0
            and choice * nested_product <= _MAX_UNROLLED_LANES
        ):
            factor = choice
    return factor


def _unroll_one(
    loop: ast.For, unroll: int, new_var: Callable[[str], str]
) -> list[ast.stmt] | None:
    extent = _ascending_range_extent(loop)
    if (
        extent is None
        or extent < 2
        or contains_compiler_marker(list(loop.body))
        or _has_scalar_policy_load(loop)
    ):
        return None
    planner = _Planner(loop)
    plan, _deps = planner.plan(loop, extent, set(), [])
    if not plan.load_count():
        return None
    factor = _unroll_factor(extent, unroll, plan.nested_product())
    if factor < 2:
        return None
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


def unroll_lane_loads(
    body: list[ast.stmt], unroll: int, new_var: Callable[[str], str]
) -> tuple[list[ast.stmt], bool]:
    """Apply the loads-first lane unroll to every lane loop reachable from
    ``body``; returns the new body and whether any loop was transformed."""
    fired = False

    def visit(statements: list[ast.stmt]) -> list[ast.stmt]:
        nonlocal fired
        result: list[ast.stmt] = []
        for stmt in statements:
            if _is_lane_loop(stmt):
                replacement = _unroll_one(stmt, unroll, new_var)  # type: ignore[arg-type]
                if replacement is not None:
                    fired = True
                    result.extend(replacement)
                    continue
            elif isinstance(stmt, (ast.For, ast.If, ast.While)):
                stmt.body = visit(stmt.body)  # type: ignore[assignment]
                stmt.orelse = visit(stmt.orelse)  # type: ignore[assignment]
            result.append(stmt)
        return result

    if unroll <= 1:
        return body, False
    return visit(body), fired
