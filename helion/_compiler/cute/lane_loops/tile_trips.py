"""The tile of a serial device loop that one thread walks under a vector row.

Role: device-body pass (root-exit lane pipeline, ahead of the owner
validation).

A tile axis on one thread walks its tile's elements in a lane loop.  Inside a
serial device loop under a trace-time vector lane loop that is

    for LANE in cutlass.range_constexpr(V):      # the vector row's lanes
        PREFIX
        for MB in range(LO, HI, U):              # one trip per tile
            for ROW in range(U):                 # the tile's rows, one thread
                INDEX = MB + cutlass.Int32(ROW)
                ...                              # LANE's reductions per row
                S = _helion_lane_reduce(X, ..., owner_lane=ROW)
                CARRY = CARRY + S                # consumes the tile's sum
        SUFFIX

where the reductions of the vector row (owned by ``LANE``) sit inside the
tile's lane loop, which the owner validation refuses ("reduction marker is
nested in a different lane owner"): the two-pass lane split of ``ROW``
cannot hold another lane's reductions.  The tile's own reductions combine no
threads, though (``threads_in_group=1``): each is a fold over the rows a
thread walks, in their order.  ``serialize_tile_lane_trips`` writes that fold
out and makes ``ROW`` a serial loop of ``MB``'s body

    for MB in range(LO, HI, U):
        HOISTED                                  # what the consumers read
        ACC = identity
        for ROW in range(U):                     # ``HELION_TILE_TRIP_ATTR``
            INDEX = MB + cutlass.Int32(ROW)
            ...
            ACC = ACC + X
        S = ACC
        CARRY = CARRY + S

which is what the lane loop computes: every statement runs once per row in
row order, the consumers of the tile's reductions once per tile after its
last row.  The lanes then carry their values through both loops in one nest
(``split.py``, ``_lane_carry_nest``), each trip of ``MB`` running ``U`` rows,
and ``cute_lane_unroll`` issues the loads of a trip's rows ahead of the first
row's reductions (``cute/unroll_lane_loads.py``).

The rewrite keeps the lane loop's result only when

* the tiles are whole: ``LO``, ``HI`` and the step are static, the step is
  the lane count and divides ``HI - LO`` (a partial last tile masks its
  lanes);
* the rows are straight-line statements, and only ``INDEX`` reads ``ROW``
  or ``MB`` (so the rows are those of one serial loop over ``LO .. HI``,
  the nest ``TileTripNest.flat`` spells out for the memory proof of the
  single nest);
* every reduction in the tile is the vector row's or a one-thread fold of
  the tile, with no other marker field set;
* the consumers of the tile's folds are plain assignments without loads
  that read, beside the folds, only values that are the same for every row
  of the tile: relocatable assignments that read no row's value, each of
  a name bound once in the tile and ahead of the consumers, which move
  ahead of the rows;
* no other statement of the rows binds a name a fold or one of its
  consumers binds: those run after the tile's last row, behind it.

Anything else keeps the refusal, with the reason appended.
"""

from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING

from ...ast_extension import expr_from_string
from ...ast_extension import statement_from_string
from ...ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from ...ast_read_writes import ReadWrites
from .._ast_pass_utils import _is_proven_relocatable_assignment
from .._ast_pass_utils import deep_clone
from .._ast_pass_utils import loop_extent
from ..fuse_two_pass_loads import _range_bounds
from .markers import _HELION_LANE_REDUCE_MARKER
from .markers import _MARKER_KEYWORD_FIELDS
from .markers import NESTED_OWNER_REFUSAL
from .markers import _combine_expr
from .markers import _dtype_ctor_from_identity
from .markers import _find_lane_reduce_call
from .markers import _is_constexpr_lane_iter
from .markers import _is_lane_reduce_marker_assign
from .markers import _LaneReduceMarker
from .unsupported import lane_loop_unsupported

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Mapping

# On the serial loop ``serialize_tile_lane_trips`` made of a tile's lane loop:
# its lane variable.  The loads-first lane unroll takes the loop as the lane
# loop it was (``cute/unroll_lane_loads.py``).
HELION_TILE_TRIP_ATTR = "_helion_tile_trip"
# On the vector lane loop around it: the ``TileTripNest``.
HELION_TILE_TRIP_NEST_ATTR = "_helion_tile_trip_nest"

# The marker fields a one-thread fold of the tile may set.  The candidate
# indices of an argmin / argmax fold like any other minimum, and keep the
# plain form of their accumulate (``_combine_expr``).
_FOLD_FIELDS = frozenset({"owner_lane", "strided_restore", "index_candidates"})


@dataclasses.dataclass
class TileTripNest:
    """What ``serialize_tile_lane_trips`` made of one vector lane loop."""

    # The serial device loop, one trip per tile.
    serial_loop: ast.For
    # The tile's rows, a serial loop of its body.
    trip_loop: ast.For
    # A copy of the vector lane loop as it stood, its serial loop walking the
    # same rows one per iteration and each fold of the tile reduced to its
    # row's value: the nest whose statements run, row by row and lane by
    # lane, exactly the memory accesses of the tiled one.
    flat: ast.For


def serialize_tile_lane_trips(
    body: list[ast.AST],
    *,
    constexpr_value: Callable[[str], int | None],
    rename_groups: Mapping[str, str],
    new_var: Callable[[str], str],
) -> None:
    """Make the one-thread tile lane loop under each vector lane loop of
    ``body`` a serial loop, in place (the module docstring has the shape and
    the conditions); raises ``BackendUnsupported`` where a nest of that shape
    does not meet them."""
    for stmt in body:
        for node in ast.walk(stmt):
            nest = _match(node)
            if nest is None:
                continue
            assert isinstance(node, ast.For)
            reason = _serialize(node, *nest, constexpr_value, rename_groups, new_var)
            if reason is not None:
                raise lane_loop_unsupported(
                    f"{NESTED_OWNER_REFUSAL}; the rows of its tile cannot run as "
                    f"a serial loop of the lanes either: {reason}",
                )


def _match(loop: ast.AST) -> tuple[ast.For, ast.For] | None:
    """The serial loop and the tile lane loop of a vector lane loop that
    holds its reductions inside a tile's lane loop, else ``None``."""
    lane_var = getattr(loop, HELION_LANE_LOOP_VAR_ATTR, None)
    if (
        not isinstance(loop, ast.For)
        or lane_var is None
        or not _is_constexpr_lane_iter(loop)
        or loop.orelse
    ):
        return None
    loops = [stmt for stmt in loop.body if isinstance(stmt, ast.For)]
    if len(loops) != 1:
        return None
    (serial,) = loops
    if (
        getattr(serial, HELION_LANE_LOOP_VAR_ATTR, None) is not None
        or _is_constexpr_lane_iter(serial)
        or serial.orelse
        or len(serial.body) != 1
    ):
        return None
    (rows,) = serial.body
    if (
        not isinstance(rows, ast.For)
        or getattr(rows, HELION_LANE_LOOP_VAR_ATTR, None) is None
        or _is_constexpr_lane_iter(rows)
    ):
        return None
    if not any(
        (marker := _is_lane_reduce_marker_assign(stmt)) is not None
        and marker.owner_lane == lane_var
        for stmt in rows.body
    ):
        return None
    return serial, rows


def _static_int(
    node: ast.expr, constexpr_value: Callable[[str], int | None]
) -> int | None:
    """A literal or a constexpr's value, through one integer cast."""
    if (
        isinstance(node, ast.Call)
        and len(node.args) == 1
        and not node.keywords
        and ast.unparse(node.func) == "cutlass.Int32"
    ):
        node = node.args[0]
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.Name):
        return constexpr_value(node.id)
    return None


def _serialize(
    loop: ast.For,
    serial: ast.For,
    rows: ast.For,
    constexpr_value: Callable[[str], int | None],
    rename_groups: Mapping[str, str],
    new_var: Callable[[str], str],
) -> str | None:
    """Rewrite one nest; the reason when it keeps its lane loop."""

    def canonical(names: Iterable[str]) -> set[str]:
        return {rename_groups.get(name, name) for name in names}

    lane_var = getattr(loop, HELION_LANE_LOOP_VAR_ATTR)
    row_var = getattr(rows, HELION_LANE_LOOP_VAR_ATTR)
    extent = loop_extent(rows, kinds=("range",))
    bounds = _range_bounds(serial.iter)
    if (
        extent is None
        or bounds is None
        or not isinstance(serial.target, ast.Name)
        or not isinstance(rows.target, ast.Name)
        or rows.target.id != row_var
    ):
        return "the tile's lanes are no loop of a literal count"
    low, high, step = (_static_int(bound, constexpr_value) for bound in bounds)
    if (
        low is None
        or high is None
        or step != extent
        or high < low
        or (high - low) % extent
    ):
        return "its serial loop does not walk whole tiles of static bounds"
    serial_var = serial.target.id
    body: list[ast.stmt] = list(rows.body)
    if any(
        isinstance(
            node,
            (
                ast.For,
                ast.While,
                ast.If,
                ast.With,
                ast.Try,
                ast.Break,
                ast.Continue,
                ast.Return,
            ),
        )
        for stmt in body
        for node in ast.walk(stmt)
    ):
        return "the tile's lanes are not straight-line statements"

    # The one statement that reads the row and tile offsets: the row index.
    index_text = f"{serial_var} + cutlass.Int32({row_var})"
    indexed = [
        stmt
        for stmt in body
        if {row_var, serial_var} & set(ReadWrites.from_ast(stmt).reads)
    ]
    if (
        len(indexed) != 1
        or not isinstance(indexed[0], ast.Assign)
        or len(indexed[0].targets) != 1
        or not isinstance(indexed[0].targets[0], ast.Name)
        or ast.unparse(indexed[0].value) != index_text
    ):
        return "the tile's lanes read their offsets beside the row index"
    (index_stmt,) = indexed

    folds: dict[int, _LaneReduceMarker] = {}
    for position, stmt in enumerate(body):
        marker = _is_lane_reduce_marker_assign(stmt)
        if marker is None:
            if _find_lane_reduce_call(stmt) is not None:
                return "a reduction of the tile is no plain assignment"
            continue
        if marker.owner_lane == lane_var:
            continue
        if (
            marker.owner_lane != row_var
            or marker.threads_in_group != 1
            or any(
                getattr(marker, field.name) != field.default
                for field in _MARKER_KEYWORD_FIELDS
                if field.name not in _FOLD_FIELDS
            )
        ):
            return "a reduction of the tile combines threads or another lane"
        folds[position] = marker
    if sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == _HELION_LANE_REDUCE_MARKER
        for stmt in body
        for node in ast.walk(stmt)
    ) != sum(_is_lane_reduce_marker_assign(stmt) is not None for stmt in body):
        return "a reduction of the tile is no plain assignment"

    # The consumers of the tile's folds, in statement order: a statement
    # ahead of a fold reads the value the tile started with.
    folded = set()
    consumers: list[int] = []
    for position, stmt in enumerate(body):
        if position in folds:
            folded |= canonical([folds[position].result_var])
            continue
        effects = ReadWrites.from_ast(stmt)
        if canonical(effects.reads) & folded:
            if not _is_proven_relocatable_assignment(stmt, allow_load=False):
                return "a consumer of the tile's reduction is no plain assignment"
            consumers.append(position)
            folded |= canonical(effects.writes)
    rest = [
        position
        for position in range(len(body))
        if position not in folds and position not in consumers
    ]
    # The folds and their consumers run after the tile's last row: a name
    # they bind is theirs alone.  A row statement that bound it as well
    # (``carry = fold`` and then ``carry = 0.0``) would run ahead of them,
    # and the consumer behind it would read the fold's value instead.
    if any(
        canonical(ReadWrites.from_ast(body[position]).writes) & folded
        for position in rest
    ):
        return "a statement of the rows binds a name the tile's reduction binds"
    # The values one row has of its own: whatever reads the row index or is
    # no pure assignment (a load, a reduction of the vector row).
    per_row = {row_var, serial_var}
    for position in rest:
        effects = ReadWrites.from_ast(body[position])
        if canonical(effects.reads) & per_row or not (
            _is_proven_relocatable_assignment(body[position], allow_load=False)
        ):
            per_row |= canonical(effects.writes)
    # What the consumers read of the tile's other statements runs once, ahead
    # of the rows.
    needed: set[str] = set()
    for position in consumers:
        needed |= canonical(ReadWrites.from_ast(body[position]).reads)
    needed -= folded
    hoisted: list[int] = []
    for position in reversed(rest):
        effects = ReadWrites.from_ast(body[position])
        written = canonical(effects.writes)
        if not written & needed:
            continue
        if written & per_row:
            return "a consumer of the tile's reduction reads a value of one row"
        hoisted.append(position)
        needed |= canonical(effects.reads)
    hoisted.reverse()
    kept = [position for position in rest if position not in hoisted]
    # A hoisted value is the one its consumers read: bound once in the tile,
    # ahead of them.
    bound: list[str] = [
        name
        for position, stmt in enumerate(body)
        if position not in folds
        for name in canonical(ReadWrites.from_ast(stmt).writes)
    ]
    if any(
        bound.count(name) != 1
        for position in hoisted
        for name in canonical(ReadWrites.from_ast(body[position]).writes)
    ) or (hoisted and consumers and hoisted[-1] > consumers[0]):
        return "a consumer of the tile's reduction reads a value bound twice"

    # The flat nest, from the statements as they stand.
    flat = deep_clone(loop)
    flat_serial = next(stmt for stmt in flat.body if isinstance(stmt, ast.For))
    (flat_rows,) = flat_serial.body
    assert isinstance(flat_rows, ast.For)
    flat_body: list[ast.stmt] = []
    for position, stmt in enumerate(flat_rows.body):
        if position in folds:
            fold = folds[position]
            flat_body.append(
                statement_from_string(
                    f"{fold.result_var} = {fold.finalize_expr(_fold_input(fold))}"
                )
            )
        elif body[position] is index_stmt:
            flat_body.append(
                statement_from_string(
                    f"{indexed[0].targets[0].id} = {serial_var} + cutlass.Int32(0)"
                )
            )
        else:
            flat_body.append(stmt)
    flat_serial.body = flat_body
    flat_bounds = _range_bounds(flat_serial.iter)
    assert flat_bounds is not None
    flat_range = expr_from_string(
        "range({low}, {high})", low=flat_bounds[0], high=flat_bounds[1]
    )
    assert isinstance(flat_range, ast.expr)
    flat_serial.iter = flat_range

    accumulators = {
        position: new_var(f"{fold.result_var}_tile_acc")
        for position, fold in folds.items()
    }
    rows.body = [
        statement_from_string(
            f"{accumulators[position]} = "
            + _combine_expr(
                folds[position].reduction_type,
                accumulators[position],
                _fold_input(folds[position]),
                _dtype_ctor_from_identity(folds[position].identity_expr),
                index_candidates=folds[position].index_candidates,
            )
        )
        if position in folds
        else body[position]
        for position in sorted([*kept, *folds])
    ]
    delattr(rows, HELION_LANE_LOOP_VAR_ATTR)
    setattr(rows, HELION_TILE_TRIP_ATTR, row_var)
    serial.body = [
        *(body[position] for position in hoisted),
        *(
            statement_from_string(f"{accumulators[position]} = {fold.identity_expr}")
            for position, fold in folds.items()
        ),
        rows,
        # A fold's value and the consumers keep their order.
        *(
            statement_from_string(
                f"{folds[position].result_var} = "
                f"{folds[position].finalize_expr(accumulators[position])}"
            )
            if position in folds
            else body[position]
            for position in sorted([*consumers, *folds])
        ),
    ]
    setattr(loop, HELION_TILE_TRIP_NEST_ATTR, TileTripNest(serial, rows, flat))
    return None


def _fold_input(fold: _LaneReduceMarker) -> str:
    """A row's value of a fold, in the accumulator's type as the lane split
    casts it."""
    ctor = _dtype_ctor_from_identity(fold.identity_expr)
    return fold.input_name if ctor is None else f"{ctor}({fold.input_name})"
