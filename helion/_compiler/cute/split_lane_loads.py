# pyrefly: ignore-errors
"""Split grid lane loops into a LOAD phase and a COMPUTE/STORE phase.

A vec-partitioned grid lane loop unrolls at trace time into

    for lane in range(U):      # U = EPT // V, constexpr
        lane_base = <affine of lane/tid/offset>
        _tile_unroll_vec_A = cute.arch.load(...lane_base..., <vec ty>)
        ...compute + store flush...

Iteration N+1's load sits AFTER iteration N's store in program order, and
ptxas cannot prove the store does not alias the load, so only ONE vector
load is in flight per thread (measured: ~5-8% DRAM-throughput deficit vs
the handwritten kernel, which loads its whole (V, U) fragment before any
compute).  Rewriting into

    _lane_loads_0 = []
    for lane in range(U):
        lane_base = ...
        _lane_loads_0.append(cute.arch.load(...))
    for lane in range(U):
        lane_base = ...
        _tile_unroll_vec_A = _lane_loads_0[lane]
        ...compute + store flush...

issues all U loads back-to-back (the trace-time Python list just names the
SSA values).  Element-order semantics match the triton backend, which
vector-loads the whole tile before any store.

Only fires when every value flowing from the load phase into the compute
phase is the lane-base or a hoisted vec load (checked), and only on loops
whose hoist vars use the ``_tile_unroll_vec_`` naming of the tile_unroll
protocol.

Both phases are trace-time copies of the lane body, so the trace (and
ptxas's time on it) grows with ``U`` copies of ``V`` elements, as do the
registers holding the U loads until their lanes compute.  The split takes at
most ``_MAX_UNROLLED_LANES`` copies and ``_MAX_UNROLLED_STATEMENTS`` traced
statements, the bounds of the knob's unroll (``cute/unroll_lane_loads.py``),
and ``_MAX_SPLIT_ELEMENTS`` elements per thread, and copies no rolled loop
other than a lane loop and a loop of one trip (``_trace_size``).  A lane loop
past these bounds stays rolled (a huge block over few threads gave
``range_constexpr(131072)``, which never finished compiling), and
``cute_lane_unroll`` unrolls it in bounded blocks unless its body holds such
a rolled loop.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ._ast_pass_utils import assigned_name
from ._ast_pass_utils import is_constexpr_range
from ._ast_pass_utils import loop_extent
from .unroll_lane_loads import _MAX_UNROLLED_LANES
from .unroll_lane_loads import _MAX_UNROLLED_STATEMENTS
from .unroll_lane_loads import _loop_trips
from .unroll_lane_loads import _trace_size

if TYPE_CHECKING:
    from collections.abc import Mapping

_LOAD_PREFIX = "_tile_unroll_vec_"
# Elements per thread (``U * V``) the two phases unroll.  Compile time grows
# faster than linearly in it (a relu/BatchNorm-backward pointwise body: 4 s at
# 256 elements, 16 s at 512, 101 s at 1024); on B200 the split of one- and
# two-input bf16 kernels ran 1.5-5x slower than the rolled loop at 1024 and
# faster or even at 512.
_MAX_SPLIT_ELEMENTS = 512


def _names_read(node: ast.AST) -> set[str]:
    out: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
            out.add(sub.id)
    return out


def _is_const_range_loop(loop: ast.For) -> tuple[str, int] | None:
    """Match ``for <lane> in range(<int >= 2>)`` (trace-time unrolled)."""
    extent = loop_extent(loop, kinds=("range",), minimum=2)
    if extent is None:
        return None
    assert isinstance(loop.target, ast.Name)
    return loop.target.id, extent


def _lane_width(
    stmts: list[ast.stmt], constexpr_values: Mapping[str, int] | None
) -> int:
    """Elements one lane computes: the most trace-time copies a statement of
    its compute phase gets, the product of the trips of the
    ``range_constexpr`` loops around it (``_loop_trips``; the V-loop's V in a
    plain lane body; 1 without one).  A range ``_loop_trips`` cannot resolve
    counts once."""
    width = 1
    for stmt in stmts:
        nested = [
            node for node in ast.iter_child_nodes(stmt) if isinstance(node, ast.stmt)
        ]
        extent = 1
        if is_constexpr_range(stmt):
            extent = _loop_trips(stmt, constexpr_values) or 1
        width = max(width, extent * _lane_width(nested, constexpr_values))
    return width


def _try_split(
    loop: ast.For, counter: int, constexpr_values: Mapping[str, int] | None
) -> list[ast.stmt] | None:
    match = _is_const_range_loop(loop)
    if match is None:
        return None
    body = loop.body
    if len(body) < 3:
        return None
    base_name = assigned_name(body[0])
    if base_name is None:
        return None
    # Collect the leading run of vec-load hoists right after the base stmt.
    load_stmts: list[ast.Assign] = []
    idx = 1
    while idx < len(body):
        name = assigned_name(body[idx])
        if name is None or not name.startswith(_LOAD_PREFIX):
            break
        load_stmts.append(body[idx])  # type: ignore[arg-type]
        idx += 1
    if not load_stmts:
        return None
    rest = body[idx:]
    if not rest:
        return None
    reps = match[1]
    statements = _trace_size(rest, constexpr_values)
    if (
        statements is None
        or reps > _MAX_UNROLLED_LANES
        or reps * _lane_width(rest, constexpr_values) > _MAX_SPLIT_ELEMENTS
        or reps * statements > _MAX_UNROLLED_STATEMENTS
    ):
        return None
    # Phase 1 writes only the lane base and the hoisted loads; both are
    # re-derived (base) or forwarded (loads) into phase 2, so no other
    # dataflow can leak across the split.
    lane_var = loop.target.id
    list_names = [f"_lane_loads_{counter}_{i}" for i in range(len(load_stmts))]
    out: list[ast.stmt] = []
    for list_name in list_names:
        out.append(statement_from_string(f"{list_name} = []"))
    # Fresh load-phase loop; the ORIGINAL load expressions move into it
    # (their statements are dropped from the compute phase), and the lane
    # base statement is duplicated via unparse (ExtendedAST nodes cannot
    # be deepcopied).
    # Both phase loops must be trace-time unrolled (range_constexpr): the
    # Python list carrying the loaded SSA values crosses loop iterations,
    # which a dynamically-lowered ``range`` loop cannot express.
    load_loop = statement_from_string(
        f"for {lane_var} in cutlass.range_constexpr({reps}):\n    pass"
    )
    assert isinstance(load_loop, ast.For)
    load_body: list[ast.stmt] = [statement_from_string(ast.unparse(body[0]))]
    for list_name, load_stmt in zip(list_names, load_stmts, strict=True):
        stmt = statement_from_string(f"{list_name}.append(_x_)")
        call = stmt.value  # type: ignore[attr-defined]
        call.args = [load_stmt.value]  # move the original load expression
        load_body.append(stmt)
    load_loop.body = load_body
    out.append(load_loop)
    compute_loop = loop
    compute_loop.iter = expr_from_string(f"cutlass.range_constexpr({reps})")
    compute_body: list[ast.stmt] = [body[0]]
    for list_name, load_stmt in zip(list_names, load_stmts, strict=True):
        load_name = assigned_name(load_stmt)
        compute_body.append(
            statement_from_string(f"{load_name} = {list_name}[{lane_var}]")
        )
    compute_body.extend(rest)
    compute_loop.body = compute_body
    out.append(compute_loop)
    return out


def split_lane_loads(
    body: list[ast.stmt], constexpr_values: Mapping[str, int] | None = None
) -> list[ast.stmt]:
    """Split each top-level grid lane loop of ``body`` that fits the bounds;
    ``constexpr_values`` resolves the bounds of rolled loops (``_trace_size``)."""
    counter = 0
    new_body: list[ast.stmt] = []
    for stmt in body:
        if isinstance(stmt, ast.For):
            split = _try_split(stmt, counter, constexpr_values)
            if split is not None:
                counter += 1
                new_body.extend(split)
                continue
            # Grid lane loops sit at the top level of the kernel body;
            # nested tile loops keep their own (rolled-reduction) passes.
        new_body.append(stmt)
    return new_body
