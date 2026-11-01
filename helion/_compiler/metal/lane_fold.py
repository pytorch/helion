"""Unroll the lane loop of a persistent dimension whose reductions fold lanes.

A persistent dimension wider than its threads is walked in a lane loop
``for L in range(E)`` that wraps the root body, tile loops included
(``MetalPersistentReductionStrategy.serialize_lanes``).  Each reduction over
it is marked ``t = _metal_lane_fold(op, v, L, T)``: before the cross-thread
combine of ``t``, every thread must combine ``v`` over all of its lanes, which
a loop running one lane at a time cannot do.  :func:`unroll_lane_folds`
replaces the loop with its body jammed across the lanes:

- a statement that depends on the lane runs once per lane, with each
  lane-dependent name ``n`` renamed ``n__lane<l>`` and ``L`` replaced by
  ``l``;
- a fold becomes the combine of its ``E`` lane values, cast to ``T``;
- every other statement runs once;
- ``for`` and ``if`` statements stay, with their bodies jammed, so the lanes
  move inside the tile loops and a loop-carried value becomes one scalar per
  lane.

This is the tile's own order.  A Helion statement acts on the whole tile, as
a Triton op acts on a whole block before the next op runs; the lane loop only
serialized the dimension, which is valid until a statement needs every lane.

Runs on the device body right before MSL emission, with no compile
environment, so the fold carries its MSL accumulator type.
"""

from __future__ import annotations

import ast
import copy
from typing import TypeVar

from .reduction import combine_expr

_Node = TypeVar("_Node", bound=ast.AST)

#: The marker ``MetalPersistentReductionStrategy.codegen_reduction`` emits.
LANE_FOLD = "_metal_lane_fold"


def unroll_lane_folds(body: list[ast.stmt]) -> list[ast.stmt]:
    """Unroll and jam every lane loop that carries a fold; *body* itself when
    it has none."""
    lanes = _fold_lanes(body)
    if not lanes:
        return body
    lane_names: set[str] = set()
    result = _unroll(body, lanes, lane_names)
    assert not _fold_lanes(result), "a lane fold outside its lane loop"
    leaked = lane_names & {
        node.id
        for stmt in result
        for node in ast.walk(stmt)
        if isinstance(node, ast.Name)
    }
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


def _fold_lanes(stmts: list[ast.stmt]) -> set[str]:
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


def _simple_statements(stmts: list[ast.stmt]) -> list[ast.stmt]:
    out: list[ast.stmt] = []
    for stmt in stmts:
        if isinstance(stmt, (ast.For, ast.If)):
            out.extend(_simple_statements(stmt.body))
            out.extend(_simple_statements(stmt.orelse))
        else:
            out.append(stmt)
    return out


def _lane_dependent(stmts: list[ast.stmt], lane: str) -> set[str]:
    """*lane* and, to a fixpoint, every name written by a statement that
    reads a lane-dependent name.  A fold's target is lane-independent."""
    dependent = {lane}
    simple = [
        stmt
        for stmt in _simple_statements(stmts)
        if not (isinstance(stmt, ast.Assign) and _is_fold(stmt.value))
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


def _check_assigned_before_read(
    stmts: list[ast.stmt], dependent: set[str], assigned: set[str]
) -> None:
    """A lane-dependent value read before the body assigns it would carry
    from one lane to the next, which a jam cannot express."""
    for stmt in stmts:
        if isinstance(stmt, (ast.For, ast.If)):
            if isinstance(stmt, ast.For):
                assigned |= _writes(stmt.target)
            _check_assigned_before_read(stmt.body, dependent, assigned)
            _check_assigned_before_read(stmt.orelse, dependent, assigned)
            continue
        carried = (_reads(stmt) & dependent) - assigned
        assert not carried, f"lane-dependent names read before assignment: {carried}"
        assigned |= _writes(stmt)


class _RenameLane(ast.NodeTransformer):
    def __init__(self, dependent: set[str], lane: str, index: int) -> None:
        self.dependent = dependent
        self.lane = lane
        self.index = index

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


def _const_extent(loop: ast.For) -> int:
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


def _jam_fold(
    stmt: ast.Assign, dependent: set[str], lane: str, extent: int
) -> ast.stmt:
    fold = stmt.value
    assert isinstance(fold, ast.Call)
    (target,) = stmt.targets
    op, value, fold_lane, acc_type = fold.args
    assert isinstance(target, ast.Name) and target.id not in dependent
    assert isinstance(fold_lane, ast.Name) and fold_lane.id == lane
    assert isinstance(op, ast.Constant) and isinstance(op.value, str)
    assert isinstance(acc_type, ast.Constant)
    parts = []
    for index in range(extent):
        lane_value = ast.unparse(_lane_copy(value, dependent, lane, index))
        # The parentheses keep each cast whole: ``static_cast<T>(a) + b``
        # would parse as one comparison chain.
        parts.append(f"(static_cast<{acc_type.value}>({lane_value}))")
    jammed = ast.parse(f"{target.id} = {balanced_combine(op.value, parts)}").body[0]
    return ast.copy_location(jammed, stmt)


def _jam(
    stmts: list[ast.stmt], dependent: set[str], lane: str, extent: int
) -> list[ast.stmt]:
    out: list[ast.stmt] = []
    for stmt in stmts:
        if isinstance(stmt, (ast.For, ast.If)):
            header = stmt.iter if isinstance(stmt, ast.For) else stmt.test
            assert not _reads(header) & dependent, (
                f"control flow on a lane-dependent value: {ast.unparse(header)}"
            )
            jammed = copy.copy(stmt)
            jammed.body = _jam(stmt.body, dependent, lane, extent)
            jammed.orelse = _jam(stmt.orelse, dependent, lane, extent)
            out.append(jammed)
        elif isinstance(stmt, ast.Assign) and _is_fold(stmt.value):
            out.append(_jam_fold(stmt, dependent, lane, extent))
        elif (_reads(stmt) | _writes(stmt)) & dependent:
            assert isinstance(stmt, (ast.Assign, ast.Expr)), ast.unparse(stmt)
            out.extend(
                _lane_copy(stmt, dependent, lane, index) for index in range(extent)
            )
        else:
            out.append(stmt)
    return out


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
            lane = stmt.target.id
            extent = _const_extent(stmt)
            dependent = _lane_dependent(stmt.body, lane)
            _check_assigned_before_read(stmt.body, dependent, {lane})
            lane_names |= dependent
            out.extend(_jam(stmt.body, dependent, lane, extent))
        elif isinstance(stmt, (ast.For, ast.If)):
            unrolled = copy.copy(stmt)
            unrolled.body = _unroll(stmt.body, lanes, lane_names)
            unrolled.orelse = _unroll(stmt.orelse, lanes, lane_names)
            out.append(unrolled)
        else:
            out.append(stmt)
    return out
