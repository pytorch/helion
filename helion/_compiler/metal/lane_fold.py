"""Unroll and jam lane loops whose lanes can't run one after another, or
whose lanes share a load.

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

:func:`jam_lanes_into_loops` jams lane loops for speed rather than validity:
one around a ``for`` loop that loads a value no lane depends on, so the load
runs once per iteration instead of once per lane.

Runs on the device body right before MSL emission, with no compile
environment, so the fold carries its MSL accumulator type.
"""

from __future__ import annotations

import ast
import copy
import math
import re
from typing import TYPE_CHECKING
from typing import TypeGuard
from typing import TypeVar
from typing import cast

from .msl_expr import cast_source
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


#: The bytes of each MSL scalar type.
MSL_SCALAR_BYTES = {
    "bool": 1,
    "char": 1,
    "uchar": 1,
    "short": 2,
    "ushort": 2,
    "half": 2,
    "bfloat": 2,
    "int": 4,
    "uint": 4,
    "float": 4,
    "long": 8,
    "ulong": 8,
}


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


#: Most copies of one statement the jams of :func:`jam_lanes_into_loops`
#: make, counting the earlier jams'.  A jam for speed copies every
#: lane-dependent statement to save the lane-independent ones; measured on
#: an M4 Pro, few-row GEMVs at 4-8 copies were 1.3-3x faster jammed, and
#: kernels with heavy per-lane work up to 1.4x slower at 64.
MAX_JAMMED_COPIES = 32

#: The suffix a jam gives a name it copies for lane ``l``: ``n__lane<l>``.
LANE_SUFFIX = "__lane"
_LANE_COPY = re.compile(rf"{LANE_SUFFIX}(\d+)")


def is_lane_loop(stmt: ast.stmt) -> TypeGuard[ast.For]:
    """A lane loop: ``lane_*`` walks a tile wider than its threads,
    ``slice_lane_*`` a persistent reduction's lanes over a ':' dim."""
    return (
        isinstance(stmt, ast.For)
        and isinstance(stmt.target, ast.Name)
        and stmt.target.id.startswith(("lane_", "slice_lane_"))
        and isinstance(stmt.iter, ast.Call)
        and isinstance(stmt.iter.func, ast.Name)
        and stmt.iter.func.id == "range"
    )


def jam_lanes_into_loops(
    body: list[ast.stmt], read_after: frozenset[str] = frozenset()
) -> list[ast.stmt]:
    """Unroll and jam each lane loop around a ``for`` loop whose body loads
    a value that doesn't depend on the lane.

    ``for L in range(E): ... for k ...: w = W[L, k]; x = X[k]; acc += w * x``
    loads ``X[k]`` (and computes whatever is derived from it) once per lane;
    jammed, the lanes move inside the ``for`` loop and that work runs once
    per iteration, with ``acc`` one scalar per lane.  A GEMV that folds rows
    into a thread, or one whose x side is computed on its loads (a gate, a
    dequantization), reads and computes x once instead of once per row.

    A jam reorders every loop in the lane loop's body, so each of them must
    have such a load: a norm's reduction loop, which loads only its row,
    runs slower with its lanes interleaved, whatever a later loop shares.

    A loop is jammed only where :class:`LaneJam` can (no control flow on a
    lane-dependent value), where no value carries from one lane to the next
    (lane-dependent or not: a jam runs a lane-independent statement once) or
    out of the loop (*read_after*: the names read after *body* runs), within
    :data:`MAX_JAMMED_COPIES` copies of a statement, and where the lane
    arrays it copies fit :data:`MAX_LANE_ARRAY_BYTES`."""
    # The names a statement of *body* leaves live: read after it before being
    # assigned again.  Worked out only beside a lane loop: a body can run to
    # thousands of statements.
    tail = ast.Expr(
        ast.Tuple([ast.Name(n, ast.Load()) for n in sorted(read_after)], ast.Load())
    )
    out: list[ast.stmt] = []
    for i, stmt in enumerate(body):
        if not any(isinstance(n, ast.For) and is_lane_loop(n) for n in ast.walk(stmt)):
            out.append(stmt)
            continue
        after = frozenset(_exposed_reads([*body[i + 1 :], tail], set()))
        if is_lane_loop(stmt) and (jam := _speed_jam(stmt, after)) is not None:
            out.extend(jam_lanes_into_loops(jam.jam(), after))
        elif isinstance(stmt, (ast.For, ast.If)):
            # A loop's next iteration reads what its body reads before
            # assigning it.
            inner = (
                after | _exposed_reads([stmt], set())
                if isinstance(stmt, ast.For)
                else after
            )
            jammed = copy.copy(stmt)
            jammed.body = jam_lanes_into_loops(stmt.body, inner)
            jammed.orelse = jam_lanes_into_loops(stmt.orelse, after)
            out.append(jammed)
        else:
            out.append(stmt)
    return out


def _speed_jam(loop: ast.For, read_after: frozenset[str]) -> LaneJam | None:
    """The jam of *loop* if :func:`jam_lanes_into_loops` takes it."""
    extent = lane_extent(loop)
    if extent is None or extent < 2:
        return None
    if extent * _copies_made(loop.body) > MAX_JAMMED_COPIES:
        return None
    jam = LaneJam(loop)
    if (
        jam.carried(_writes(loop) | jam.dependent)
        or jam.dependent & read_after
        or _branches_on(loop.body, jam.dependent)
        or not _every_loop_shares_a_load(loop.body, jam.dependent)
        or extent * _lane_array_bytes(loop.body, jam.dependent) > MAX_LANE_ARRAY_BYTES
    ):
        return None
    return jam


def _lane_array_bytes(stmts: list[ast.stmt], dependent: set[str]) -> int:
    """The bytes of the lane arrays in *stmts* a jam copies per lane: those
    the lanes write.  Copies past :data:`MAX_LANE_ARRAY_BYTES` per thread
    cost more than the jam saves."""
    total = 0
    for stmt in stmts:
        for node in ast.walk(stmt):
            if (
                isinstance(node, ast.Assign)
                and isinstance(call := node.value, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "_metal_lane_array"
                and any(_writes(t) & dependent for t in node.targets)
            ):
                dtype, length = (cast("ast.Constant", a).value for a in call.args)
                assert isinstance(dtype, str) and isinstance(length, int)
                total += MSL_SCALAR_BYTES[dtype] * length
    return total


def _copies_made(stmts: list[ast.stmt]) -> int:
    """An estimate of the most copies of one statement the earlier jams of
    *stmts* made: one they renamed ``n__lane<l>`` has ``l + 1`` or more per
    jam (a copy that only took its lane as a constant has no such name)."""
    best = 1
    for stmt in stmts:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name):
                copies = math.prod(int(i) + 1 for i in _LANE_COPY.findall(node.id))
                best = max(best, copies)
    return best


def _branches_on(stmts: list[ast.stmt], dependent: set[str]) -> bool:
    for stmt in stmts:
        if isinstance(stmt, (ast.For, ast.If)):
            header = stmt.iter if isinstance(stmt, ast.For) else stmt.test
            if _reads(header) & dependent or _branches_on(
                stmt.body + stmt.orelse, dependent
            ):
                return True
    return False


def _every_loop_shares_a_load(stmts: list[ast.stmt], dependent: set[str]) -> bool:
    """Whether *stmts* hold a ``for`` loop, and each outermost one loads
    something without touching a lane-dependent name.  An inner lane loop is
    no such loop: what it would share is an element already broadcast to its
    lanes."""
    loops: list[ast.For] = []

    def outermost(stmts: list[ast.stmt]) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.For) and not is_lane_loop(stmt):
                loops.append(stmt)
            elif isinstance(stmt, (ast.For, ast.If)):
                outermost(stmt.body + stmt.orelse)

    def shares(stmt: ast.stmt) -> bool:
        if isinstance(stmt, (ast.For, ast.If)):
            return any(map(shares, stmt.body + stmt.orelse))
        return not (_reads(stmt) | _writes(stmt)) & dependent and any(
            tl_load(n) for n in ast.walk(stmt)
        )

    outermost(stmts)
    return bool(loops) and all(any(map(shares, loop.body)) for loop in loops)


def tl_load(node: ast.AST) -> ast.Call | None:
    """*node* if it is a ``tl.load`` call: a read of device memory."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "tl"
        and node.func.attr == "load"
    ):
        return node
    return None


def load_mask_other(load: ast.Call) -> tuple[ast.expr, ast.expr | None] | None:
    """The mask and fill value (None if it has none) of
    ``tl.load(ptr, mask, other)``, or None for an unmasked load."""
    if len(load.args) < 2:
        return None
    mask = load.args[1]
    if isinstance(mask, ast.Constant) and mask.value is None:
        return None
    other = load.args[2] if len(load.args) > 2 else None
    return mask, next((k.value for k in load.keywords if k.arg == "other"), other)


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


#: The element store into a lane array (``lane_reduction``'s kept loads,
#: ``mpp_matmul``'s staged loads): ``_metal_lane_put(array, index, value)``.
LANE_PUT = "_metal_lane_put"


def _array_writes(tree: ast.AST, lane: str) -> set[str]:
    """The lane arrays *tree* stores into at an index that doesn't read
    *lane*.  Such an array is one per lane once a stored value depends on
    the lane: every lane's copy of the store writes the same element.  An
    array indexed by *lane* itself has one element per lane and stays
    shared."""
    return {
        node.args[0].id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == LANE_PUT
        and isinstance(node.args[0], ast.Name)
        and lane not in _reads(node.args[1])
    }


#: The integer-rounded float -> bfloat conversion every lane copy uses
#: (``metal_jit`` emits its definition when it is referenced).
BF16_CAST = "helion_bf16_cast"


class _RenameLane(ast.NodeTransformer):
    def __init__(self, dependent: set[str], lane: str, index: int) -> None:
        self.dependent = dependent
        self.lane = lane
        self.index = index

    def visit_Call(self, node: ast.Call) -> ast.AST:
        # Metal miscompiles float -> bfloat conversions that lane copies
        # multiply (exact in float16 and float32 at the same MSL); round
        # them in integer arithmetic, as the MPP path's float tiles do.
        self.generic_visit(node)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "static_cast"
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "bfloat"
        ):
            return ast.copy_location(
                ast.Call(
                    func=ast.Name(id=BF16_CAST, ctx=ast.Load()),
                    args=node.args[1:],
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
                ast.Name(id=f"{node.id}{LANE_SUFFIX}{self.index}", ctx=node.ctx), node
            )
        return node


def _lane_copy(node: _Node, dependent: set[str], lane: str, index: int) -> _Node:
    return _RenameLane(dependent, lane, index).visit(copy.deepcopy(node))


def lane_extent(loop: ast.For) -> int | None:
    """The constant trip count of the lane loop *loop*, if it has one."""
    loop_iter = loop.iter
    if (
        isinstance(loop_iter, ast.Call)
        and isinstance(loop_iter.func, ast.Name)
        and loop_iter.func.id == "range"
        and len(loop_iter.args) == 1
        and isinstance(loop_iter.args[0], ast.Constant)
        and isinstance(loop_iter.args[0].value, int)
    ):
        return loop_iter.args[0].value
    return None


def const_extent(loop: ast.For) -> int:
    """The constant trip count of the lane loop *loop*."""
    extent = lane_extent(loop)
    assert extent is not None, (
        f"lane loop without a constant extent: {ast.unparse(loop.iter)}"
    )
    return extent


def _never(stmt: ast.stmt) -> set[str] | None:
    return None


def _exposed_reads(
    stmts: list[ast.stmt],
    assigned: set[str],
    unit_reads: Callable[[ast.stmt], set[str] | None] = _never,
) -> set[str]:
    """The names *stmts* read where they haven't definitely assigned them
    (nor are they in *assigned*); *unit_reads* gives what a statement run
    whole reads, None for one to walk.  An ``if`` assigns what both
    branches assign, a ``for`` body nothing past the loop: MSL scopes a
    declaration to its block, so a value assigned only there is gone after
    it."""
    found: set[str] = set()

    def walk(stmts: list[ast.stmt], assigned: set[str]) -> set[str]:
        assigned = set(assigned)
        for stmt in stmts:
            reads = unit_reads(stmt)
            if isinstance(stmt, ast.For) and reads is None:
                found.update(_reads(stmt.iter) - assigned)
                walk(stmt.body, assigned | _writes(stmt.target))
            elif isinstance(stmt, ast.If) and reads is None:
                found.update(_reads(stmt.test) - assigned)
                assigned |= walk(stmt.body, assigned) & walk(stmt.orelse, assigned)
            else:
                found.update((_reads(stmt) if reads is None else reads) - assigned)
                assigned |= _writes(stmt)
        return assigned

    walk(stmts, assigned)
    return found


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
        that reads a lane-dependent name, a lane array stored into included
        (see ``_array_writes``).  A unit's targets are lane-independent."""
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
                written = _writes(stmt) | _array_writes(stmt, self.lane)
                if written - dependent and _reads(stmt) & dependent:
                    dependent |= written
                    changed = True
        return dependent

    def carried(self, names: set[str] | None = None) -> set[str]:
        """The lane-dependent names (or *names*) read where the body hasn't
        definitely assigned them: each would carry from one lane to the
        next, which a copy per lane can't hold."""
        tracked = self.dependent if names is None else names
        return _exposed_reads(self.loop.body, {self.lane}, self._unit_reads) & tracked

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
        assert isinstance(acc_type, ast.Constant) and isinstance(acc_type.value, str)
        parts = []
        for index in range(self.extent):
            lane_value = ast.unparse(
                _lane_copy(value, self.dependent, self.lane, index)
            )
            parts.append(cast_source(acc_type.value, lane_value))
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
