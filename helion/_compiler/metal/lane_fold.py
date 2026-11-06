"""Unroll and jam lane loops whose lanes can't run one after another.

A block wider than its threads is walked in a lane loop ``for L in range(E)``
around the statements over it.  That is valid while no statement needs
another lane's work of the same iteration.  Two kinds of statement do:

- a reduction over a persistent dimension wider than its threads
  (``MetalPersistentReductionStrategy.serialize_lanes``), marked
  ``t = _metal_lane_fold(op, v, L, T)``: before the cross-thread combine of
  ``t``, every thread must combine ``v`` over all of its lanes;
- a cooperative MPP op (``mpp_matmul``), which reads every lane's staged
  elements and must run once.

:class:`LaneJam` replaces such a loop with its body jammed across the lanes:

- a statement that depends on the lane runs once per lane, with each
  lane-dependent name ``n`` renamed ``n__lane<l>`` and ``L`` replaced by
  ``l``;
- a fold becomes the combine of its ``E`` lane values, cast to ``T``;
- a statement the caller runs once (a cooperative MPP statement) and every
  other statement run once;
- ``for`` and ``if`` statements stay, with their bodies jammed, so the lanes
  move inside the tile loops and a loop-carried value becomes one scalar per
  lane.

This is the tile's own order.  A Helion statement acts on the whole tile, as
a Triton op acts on a whole block before the next op runs; the lane loop only
serialized the block, which is valid until a statement needs every lane.

Runs on the device body right before MSL emission, with no compile
environment, so the fold carries its MSL accumulator type.
"""

from __future__ import annotations

import ast
import copy
from typing import TYPE_CHECKING
from typing import TypeVar

from .reduction import combine_expr

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

_Node = TypeVar("_Node", bound=ast.AST)

#: The marker ``MetalPersistentReductionStrategy.codegen_reduction`` emits.
LANE_FOLD = "_metal_lane_fold"

#: Most copies of one statement a thread's jams make.  Each copy keeps its own
#: values, so code size and register pressure grow with the count.  The bound
#: is CuTe's for an unrolled per-thread register tile
#: (``CUTE_REGISTER_TILE_MAX_ELEMENTS``).
MAX_LANE_COPIES = 64

#: Most bytes of lane arrays one split keeps per thread.  Measured on Apple
#: GPUs: arrays of 16-bit loads up to 64 bytes were neutral or faster than
#: reloading, 128 bytes mixed and 256 bytes up to 2.3x slower; 32-bit arrays
#: were neutral to 5% slower at any size, so only narrower loads are kept.
MAX_LANE_ARRAY_BYTES = 64


class LaneArrayBudget:
    """The per-thread lane arrays of one split, ``lanes`` elements each."""

    def __init__(self, lanes: int) -> None:
        self.lanes = lanes
        self.left = MAX_LANE_ARRAY_BYTES

    def take(self, dtype: torch.dtype) -> bool:
        """Whether a load of ``dtype`` gets an array, charging it if so."""
        size = self.lanes * dtype.itemsize
        if dtype.itemsize >= 4 or size > self.left:
            return False
        self.left -= size
        return True


def unroll_lane_folds(
    body: list[ast.stmt], reads: Callable[[ast.stmt], set[str]]
) -> list[ast.stmt]:
    """Unroll and jam every lane loop that carries a fold; *body* itself when
    it has none.  *reads* gives the names a statement reads, which for an
    MPP element loop excludes the names it binds and defines itself."""
    lanes = folded_lanes(body)
    if not lanes:
        return body
    lane_names: set[str] = set()
    result = _unroll(body, lanes, lane_names)
    assert not folded_lanes(result), "a lane fold outside its lane loop"
    leaked = lane_names & set().union(*map(reads, result))
    assert not leaked, f"lane-dependent names used outside their lane loop: {leaked}"
    return result


def balanced_combine(reduction_type: str, parts: list[str]) -> str:
    """Combine *parts* pairwise, level by level, so nesting grows with the
    log of the lane count; an odd part is carried up to the next level."""
    while len(parts) > 1:
        pairs = [
            combine_expr(reduction_type, lhs, rhs)
            for lhs, rhs in zip(parts[::2], parts[1::2], strict=False)
        ]
        parts = pairs + parts[2 * len(pairs) :]
    return parts[0]


def _is_fold(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == LANE_FOLD
    )


def _is_fold_stmt(stmt: ast.stmt) -> bool:
    return isinstance(stmt, ast.Assign) and _is_fold(stmt.value)


def folded_lanes(stmts: list[ast.stmt]) -> set[str]:
    lanes: set[str] = set()
    for stmt in stmts:
        for node in ast.walk(stmt):
            if _is_fold(node):
                assert isinstance(node, ast.Call)
                lane = node.args[2]
                assert isinstance(lane, ast.Name)
                lanes.add(lane.id)
    return lanes


def _reads(node: ast.AST) -> set[str]:
    """Names *node* reads.  A fold reads only its value: its result is the
    same for every lane."""
    found: set[str] = set()
    pending = [node]
    while pending:
        current = pending.pop()
        if _is_fold(current):
            assert isinstance(current, ast.Call)
            pending.append(current.args[1])
        elif isinstance(current, ast.Name):
            if isinstance(current.ctx, ast.Load):
                found.add(current.id)
        else:
            pending.extend(ast.iter_child_nodes(current))
    return found


def _writes(tree: ast.AST) -> set[str]:
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }


#: The integer-rounded float -> bfloat conversion every lane copy uses
#: (``metal_jit`` emits its definition when it is referenced).
BF16_CAST = "helion_bf16_cast"


class _RenameLane(ast.NodeTransformer):
    def __init__(self, dependent: set[str], lane: str, index: int) -> None:
        self.dependent = dependent
        self.lane = lane
        self.index = index

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        # Metal miscompiles float -> bfloat conversions that lane copies
        # multiply (exact in float16 and float32 at the same MSL); round
        # them in integer arithmetic, as the MPP path's float tiles do.
        self.generic_visit(node)
        if (
            isinstance(node.left, ast.Name)
            and node.left.id == "static_cast"
            and len(node.ops) == 2
            and isinstance(node.ops[0], ast.Lt)
            and isinstance(node.ops[1], ast.Gt)
            and isinstance(node.comparators[0], ast.Name)
            and node.comparators[0].id == "bfloat"
        ):
            return ast.copy_location(
                ast.Call(
                    func=ast.Name(id=BF16_CAST, ctx=ast.Load()),
                    args=[node.comparators[1]],
                    keywords=[],
                ),
                node,
            )
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == self.lane:
            return ast.copy_location(ast.Constant(value=self.index), node)
        if node.id in self.dependent:
            return ast.copy_location(
                ast.Name(id=f"{node.id}__lane{self.index}", ctx=node.ctx), node
            )
        return node


def _lane_copy(node: _Node, dependent: set[str], lane: str, index: int) -> _Node:
    return _RenameLane(dependent, lane, index).visit(copy.deepcopy(node))


def const_extent(loop: ast.For) -> int:
    """The constant trip count of the lane loop *loop*."""
    loop_iter = loop.iter
    assert (
        isinstance(loop_iter, ast.Call)
        and isinstance(loop_iter.func, ast.Name)
        and loop_iter.func.id == "range"
        and len(loop_iter.args) == 1
        and isinstance(loop_iter.args[0], ast.Constant)
        and isinstance(loop_iter.args[0].value, int)
    ), f"lane loop without a constant extent: {ast.unparse(loop_iter)}"
    return loop_iter.args[0].value


def _never(stmt: ast.stmt) -> set[str] | None:
    return None


class LaneJam:
    """The body of the lane loop *loop*, jammed across its lanes.

    *run_once* returns the names a statement reads when the caller runs it
    once as a whole, and None for any other statement.  The jam neither
    looks inside such a statement nor copies it, and its targets are
    lane-independent; :meth:`lane_reading_units` lists the ones that read a
    lane-dependent name anyway, which the caller must reject.
    """

    def __init__(
        self,
        loop: ast.For,
        run_once: Callable[[ast.stmt], set[str] | None] = _never,
    ) -> None:
        assert isinstance(loop.target, ast.Name)
        self.loop = loop
        self.lane = loop.target.id
        self.extent = const_extent(loop)
        self.run_once = run_once
        self.dependent = self._lane_dependent()

    def _unit_reads(self, stmt: ast.stmt) -> set[str] | None:
        if _is_fold_stmt(stmt):
            return _reads(stmt)
        return self.run_once(stmt)

    def _simple(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
        out: list[ast.stmt] = []
        for stmt in stmts:
            if isinstance(stmt, (ast.For, ast.If)) and self._unit_reads(stmt) is None:
                out.extend(self._simple(stmt.body))
                out.extend(self._simple(stmt.orelse))
            else:
                out.append(stmt)
        return out

    def _lane_dependent(self) -> set[str]:
        """The lane and, to a fixpoint, every name written by a statement
        that reads a lane-dependent name.  A unit's targets are
        lane-independent."""
        dependent = {self.lane}
        simple = [
            stmt
            for stmt in self._simple(self.loop.body)
            if self._unit_reads(stmt) is None
        ]
        changed = True
        while changed:
            changed = False
            for stmt in simple:
                written = _writes(stmt)
                if written - dependent and _reads(stmt) & dependent:
                    dependent |= written
                    changed = True
        return dependent

    def carried(self) -> set[str]:
        """The lane-dependent names read where the body hasn't definitely
        assigned them: each would carry from one lane to the next, which a
        copy per lane can't hold.  An ``if`` assigns what both branches
        assign, a ``for`` body nothing past the loop: MSL scopes a
        declaration to its block, so a copy assigned only there is gone
        after it."""
        found: set[str] = set()

        def walk(stmts: list[ast.stmt], assigned: set[str]) -> set[str]:
            assigned = set(assigned)
            for stmt in stmts:
                reads = self._unit_reads(stmt)
                if isinstance(stmt, ast.For) and reads is None:
                    found.update((_reads(stmt.iter) & self.dependent) - assigned)
                    walk(stmt.body, assigned | _writes(stmt.target))
                elif isinstance(stmt, ast.If) and reads is None:
                    found.update((_reads(stmt.test) & self.dependent) - assigned)
                    assigned |= walk(stmt.body, assigned) & walk(stmt.orelse, assigned)
                else:
                    found.update(
                        ((_reads(stmt) if reads is None else reads) & self.dependent)
                        - assigned
                    )
                    assigned |= _writes(stmt)
            return assigned

        walk(self.loop.body, {self.lane})
        return found

    def lane_reading_units(self) -> list[ast.stmt]:
        """The caller's run-once statements that read a lane-dependent name."""
        found: list[ast.stmt] = []

        def walk(stmts: list[ast.stmt]) -> None:
            for stmt in stmts:
                if _is_fold_stmt(stmt):
                    continue
                reads = self.run_once(stmt)
                if reads is not None:
                    if reads & self.dependent:
                        found.append(stmt)
                elif isinstance(stmt, (ast.For, ast.If)):
                    walk(stmt.body)
                    walk(stmt.orelse)

        walk(self.loop.body)
        return found

    def jam(self) -> list[ast.stmt]:
        """The jammed body; the names it renamed are ``dependent``."""
        return self._jam(self.loop.body)

    def _jam(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
        out: list[ast.stmt] = []
        for stmt in stmts:
            if _is_fold_stmt(stmt):
                assert isinstance(stmt, ast.Assign)
                out.append(self._jam_fold(stmt))
            elif self.run_once(stmt) is not None:
                out.append(stmt)
            elif isinstance(stmt, (ast.For, ast.If)):
                header = stmt.iter if isinstance(stmt, ast.For) else stmt.test
                assert not _reads(header) & self.dependent, (
                    f"control flow on a lane-dependent value: {ast.unparse(header)}"
                )
                jammed = copy.copy(stmt)
                jammed.body = self._jam(stmt.body)
                jammed.orelse = self._jam(stmt.orelse)
                out.append(jammed)
            elif (_reads(stmt) | _writes(stmt)) & self.dependent:
                assert isinstance(stmt, (ast.Assign, ast.Expr)), ast.unparse(stmt)
                out.extend(
                    _lane_copy(stmt, self.dependent, self.lane, index)
                    for index in range(self.extent)
                )
            else:
                out.append(stmt)
        return out

    def _jam_fold(self, stmt: ast.Assign) -> ast.stmt:
        fold = stmt.value
        assert isinstance(fold, ast.Call)
        (target,) = stmt.targets
        op, value, fold_lane, acc_type = fold.args
        assert isinstance(target, ast.Name) and target.id not in self.dependent
        assert isinstance(fold_lane, ast.Name) and fold_lane.id == self.lane
        assert isinstance(op, ast.Constant) and isinstance(op.value, str)
        assert isinstance(acc_type, ast.Constant)
        parts = []
        for index in range(self.extent):
            lane_value = ast.unparse(
                _lane_copy(value, self.dependent, self.lane, index)
            )
            # The parentheses keep each cast whole: ``static_cast<T>(a) + b``
            # would parse as one comparison chain.
            parts.append(f"(static_cast<{acc_type.value}>({lane_value}))")
        jammed = ast.parse(f"{target.id} = {balanced_combine(op.value, parts)}").body[0]
        return ast.copy_location(jammed, stmt)


def _unroll(
    stmts: list[ast.stmt], lanes: set[str], lane_names: set[str]
) -> list[ast.stmt]:
    out: list[ast.stmt] = []
    for stmt in stmts:
        if (
            isinstance(stmt, ast.For)
            and isinstance(stmt.target, ast.Name)
            and stmt.target.id in lanes
        ):
            jam = LaneJam(stmt)
            carried = jam.carried()
            assert not carried, (
                f"lane-dependent names read before assignment: {carried}"
            )
            lane_names |= jam.dependent
            out.extend(jam.jam())
        elif isinstance(stmt, (ast.For, ast.If)):
            unrolled = copy.copy(stmt)
            unrolled.body = _unroll(stmt.body, lanes, lane_names)
            unrolled.orelse = _unroll(stmt.orelse, lanes, lane_names)
            out.append(unrolled)
        else:
            out.append(stmt)
    return out
