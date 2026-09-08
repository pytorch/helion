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
"""

from __future__ import annotations

import ast

from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string

_LOAD_PREFIX = "_tile_unroll_vec_"


def _names_read(node: ast.AST) -> set[str]:
    out: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
            out.add(sub.id)
    return out


def _assigned_name(stmt: ast.stmt) -> str | None:
    if (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
    ):
        return stmt.targets[0].id
    return None


def _is_const_range_loop(loop: ast.For) -> tuple[str, int] | None:
    """Match ``for <lane> in range(<int >= 2>)`` (trace-time unrolled)."""
    if not (isinstance(loop.target, ast.Name) and isinstance(loop.iter, ast.Call)):
        return None
    func = loop.iter.func
    if not (isinstance(func, ast.Name) and func.id == "range"):
        return None
    if len(loop.iter.args) != 1 or loop.iter.keywords:
        return None
    arg = loop.iter.args[0]
    if not (isinstance(arg, ast.Constant) and isinstance(arg.value, int)):
        return None
    if arg.value < 2:
        return None
    return loop.target.id, arg.value


def _try_split(loop: ast.For, counter: int) -> list[ast.stmt] | None:
    match = _is_const_range_loop(loop)
    if match is None:
        return None
    body = loop.body
    if len(body) < 3:
        return None
    base_name = _assigned_name(body[0])
    if base_name is None:
        return None
    # Collect the leading run of vec-load hoists right after the base stmt.
    load_stmts: list[ast.Assign] = []
    idx = 1
    while idx < len(body):
        name = _assigned_name(body[idx])
        if name is None or not name.startswith(_LOAD_PREFIX):
            break
        load_stmts.append(body[idx])  # type: ignore[arg-type]
        idx += 1
    if not load_stmts:
        return None
    rest = body[idx:]
    if not rest:
        return None
    # Phase 1 writes only the lane base and the hoisted loads; both are
    # re-derived (base) or forwarded (loads) into phase 2, so no other
    # dataflow can leak across the split.
    lane_var = loop.target.id
    reps = match[1]
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
        load_name = _assigned_name(load_stmt)
        compute_body.append(
            statement_from_string(f"{load_name} = {list_name}[{lane_var}]")
        )
    compute_body.extend(rest)
    compute_loop.body = compute_body
    out.append(compute_loop)
    return out


def split_lane_loads(body: list[ast.stmt]) -> list[ast.stmt]:
    counter = 0
    new_body: list[ast.stmt] = []
    for stmt in body:
        if isinstance(stmt, ast.For):
            split = _try_split(stmt, counter)
            if split is not None:
                counter += 1
                new_body.extend(split)
                continue
            # Grid lane loops sit at the top level of the kernel body;
            # nested tile loops keep their own (rolled-reduction) passes.
        new_body.append(stmt)
    return new_body
