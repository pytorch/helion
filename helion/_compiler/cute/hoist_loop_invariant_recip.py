"""AST peephole that hoists ``x / scalar`` divisions out of inner loops
when the divisor is loop-invariant (transitively).

The CuTe softmax two-pass kernel computes ``out[k] = exp(x[k] - mi) / di``
in its second pass.  ``di`` is a single fp32 scalar per row that doesn't
change across the inner tile loop, but each inner iteration emits a
``v_12 = v_11 / di_copy_1_0`` floating-point divide.  Divides are slow
(~22 cycles on B200 vs 2 cycles for a multiply), and softmax fp16/bf16
shapes (e.g. (4096, 12672)) end up doing ~12672 divides per row per CTA.

This pass detects divisions inside for-loops where the DIVISOR is
loop-invariant — even when the divisor is a chain of copy
assignments (``di_copy_1 = di; di_copy_1_0 = di_copy_1``) since the
Helion ``ast_extension`` passes routinely insert such copies for SSA
maintenance.

A name is considered loop-invariant if:
  - It is read but never written inside the loop body, OR
  - It IS written inside the loop, but ONLY by Assign statements whose
    RHS is itself loop-invariant (transitively).

The rewrite then hoists the reciprocal computation:

    INV_DIVISOR = 1.0 / DIVISOR         # hoisted above the loop
    for OFFSET in range(...):
        ...
        QUOT = NUMERATOR * INV_DIVISOR
        ...

Multiple divisions sharing the same loop-invariant divisor share a
single reciprocal computation.

Measured impact on (4096, 12672) fp16 softmax_two_pass on B200:
  Before: 140us / 1486 GB/s
  After:  117us / 1776 GB/s   (+20% bench gain)
"""

from __future__ import annotations

import ast

from ..ast_extension import statement_from_string

_HOIST_COUNTER: list[int] = [0]


def _new_inv_name() -> str:
    name = f"_helion_inv_div_{_HOIST_COUNTER[0]}"
    _HOIST_COUNTER[0] += 1
    return name


class _NameRefCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.names.add(node.id)


def _names_read(node: ast.AST) -> set[str]:
    collector = _NameRefCollector()
    collector.visit(node)
    return collector.names


def _collect_assigns_in_body(
    body: list[ast.stmt],
) -> tuple[dict[str, list[ast.expr]], set[str]]:
    """Walk ``body`` recursively and collect:
      - assigns_by_target: dict[name -> list of RHS expressions assigned]
      - all_written: set of all names that appear as Assign target anywhere
                     (including augmented assigns and tuple targets,
                     conservatively).

    Returns both so callers can do transitive invariance analysis.
    """
    assigns_by_target: dict[str, list[ast.expr]] = {}
    all_written: set[str] = set()

    def _walk(stmts: list[ast.stmt]) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        all_written.add(target.id)
                        assigns_by_target.setdefault(target.id, []).append(stmt.value)
                    elif isinstance(target, (ast.Tuple, ast.List)):
                        # Conservatively mark as non-invariant: we can't
                        # easily reason about destructuring.
                        for elt in target.elts:
                            if isinstance(elt, ast.Name):
                                all_written.add(elt.id)
                                # Don't add to assigns_by_target -> treated
                                # as non-invariant.
            elif isinstance(stmt, ast.AugAssign):
                target = stmt.target
                if isinstance(target, ast.Name):
                    all_written.add(target.id)
                    # AugAssign reads + writes the same name; never invariant.
            # Recurse into nested for / if / with.
            if isinstance(stmt, (ast.For, ast.If, ast.With)):
                _walk(stmt.body)
            if isinstance(stmt, (ast.For, ast.If)):
                _walk(stmt.orelse)

    _walk(body)
    return assigns_by_target, all_written


def _is_loop_invariant(
    name: str,
    assigns_by_target: dict[str, list[ast.expr]],
    all_written: set[str],
    loop_target: str | None,
    invariant_cache: dict[str, bool],
    visiting: set[str],
) -> bool:
    """Return True iff ``name`` is loop-invariant.

    Definitions:
      - The loop's induction variable (``loop_target``) is NEVER invariant.
      - A name not written in the loop is invariant (assumed defined
        outside the loop and not mutated).
      - A name written exactly N times, where ALL RHS values are
        invariant, is invariant.
      - All other names (multiple defs with non-invariant RHS, augassigns,
        tuple-destructured, etc.) are non-invariant.

    Uses memoization to avoid infinite recursion on cycles (cycles are
    treated as non-invariant).
    """
    if name == loop_target:
        return False
    if name in invariant_cache:
        return invariant_cache[name]
    if name in visiting:
        # Cycle — conservatively non-invariant.
        invariant_cache[name] = False
        return False
    if name not in all_written:
        # Defined outside the loop, not mutated inside — invariant.
        invariant_cache[name] = True
        return True
    rhs_list = assigns_by_target.get(name)
    if rhs_list is None:
        # Was written via tuple destructure or augassign — conservatively
        # non-invariant.
        invariant_cache[name] = False
        return False

    visiting.add(name)
    invariant = True
    for rhs in rhs_list:
        # Loop-invariant if all Name reads in rhs are invariant AND rhs
        # contains no non-Name, non-Constant subexpressions referring to
        # the loop target.
        # First, conservatively forbid Subscript / Call / BinOp that we
        # can't fully analyze.  Actually, allow them — the deciding
        # factor is whether the rhs reads any non-invariant name.
        rhs_reads = _names_read(rhs)
        for r in rhs_reads:
            if not _is_loop_invariant(
                r,
                assigns_by_target,
                all_written,
                loop_target,
                invariant_cache,
                visiting,
            ):
                invariant = False
                break
        if not invariant:
            break
    visiting.discard(name)
    invariant_cache[name] = invariant
    return invariant


def _resolve_root_name(
    name: str,
    assigns_by_target: dict[str, list[ast.expr]],
    all_written: set[str],
    visited: set[str],
) -> str | None:
    """Trace a chain of single-assignment ``x_copy = x_root`` aliases back
    to the root name defined OUTSIDE the loop.

    Returns the outermost name (one not in ``all_written``) or None when
    the chain can't be cleanly resolved (multiple defs, non-Name RHS,
    cycle, ...).
    """
    if name in visited:
        return None
    if name not in all_written:
        return name
    rhs_list = assigns_by_target.get(name)
    if rhs_list is None or len(rhs_list) != 1:
        return None
    rhs = rhs_list[0]
    if not isinstance(rhs, ast.Name):
        return None
    visited.add(name)
    return _resolve_root_name(rhs.id, assigns_by_target, all_written, visited)


def _find_invariant_div_binops(
    loop_body: list[ast.stmt],
    loop_target: str | None,
) -> dict[str, list[ast.BinOp]]:
    """Walk loop_body and find ``x / DIVISOR`` BinOp nodes where DIVISOR
    is loop-invariant.  Maps the loop-EXTERNAL root divisor-name ->
    list of BinOp nodes.

    Only considers float-style divides (ast.Div), not floor-divides.
    """
    assigns_by_target, all_written = _collect_assigns_in_body(loop_body)
    invariant_cache: dict[str, bool] = {}
    found: dict[str, list[ast.BinOp]] = {}
    for stmt in loop_body:
        for sub in ast.walk(stmt):
            if (
                isinstance(sub, ast.BinOp)
                and isinstance(sub.op, ast.Div)
                and isinstance(sub.right, ast.Name)
            ):
                divisor_name = sub.right.id
                if not _is_loop_invariant(
                    divisor_name,
                    assigns_by_target,
                    all_written,
                    loop_target,
                    invariant_cache,
                    set(),
                ):
                    continue
                # Resolve the divisor to its loop-EXTERNAL root name so
                # the hoisted reciprocal computation can reference a name
                # visible outside the loop.
                root = _resolve_root_name(
                    divisor_name, assigns_by_target, all_written, set()
                )
                if root is None:
                    continue
                found.setdefault(root, []).append(sub)
    return found


def _rewrite_div_to_mul(node: ast.BinOp, inv_name: str) -> None:
    """In-place rewrite of ``x / DIVISOR`` to ``x * INV_NAME``."""
    node.op = ast.Mult()
    node.right = ast.Name(id=inv_name, ctx=ast.Load())
    ast.fix_missing_locations(node)


def _hoist_invariant_recips_in_for(
    for_node: ast.For,
    parent_body: list[ast.stmt],
    for_idx: int,
) -> int:
    """Try to hoist loop-invariant divisions in a single for-loop.

    Returns the number of statements inserted before ``for_idx`` (so the
    caller can adjust subsequent indices).
    """
    loop_target = for_node.target.id if isinstance(for_node.target, ast.Name) else None
    divisor_map = _find_invariant_div_binops(for_node.body, loop_target)
    if not divisor_map:
        return 0

    inserted = 0
    for divisor_name, binop_list in divisor_map.items():
        inv_name = _new_inv_name()
        decl = statement_from_string(f"{inv_name} = 1.0 / {divisor_name}")
        parent_body.insert(for_idx + inserted, decl)
        inserted += 1
        for binop in binop_list:
            _rewrite_div_to_mul(binop, inv_name)
    return inserted


def _hoist_in_body(body: list[ast.stmt]) -> list[ast.stmt]:
    """Walk ``body`` recursively.  For each for-loop at this level, look
    for loop-invariant divisions and hoist the reciprocal computation.

    Returns the (possibly modified) body list — mutates in place when
    hoists fire.  Recurses depth-first so inner loops fire first.
    """
    # First recurse into nested bodies (so the inner-most loops get
    # rewritten before outer ones).
    for stmt in body:
        if isinstance(stmt, (ast.For, ast.If, ast.With, ast.FunctionDef)):
            stmt.body = _hoist_in_body(stmt.body)
        if isinstance(stmt, (ast.For, ast.If)):
            stmt.orelse = _hoist_in_body(stmt.orelse)

    # Now walk this level looking for for-loops to hoist out of.  Insert
    # decls BEFORE the for-loop in this body.
    i = 0
    while i < len(body):
        stmt = body[i]
        if isinstance(stmt, ast.For) and not stmt.orelse:
            inserted = _hoist_invariant_recips_in_for(stmt, body, i)
            # Skip past the inserted decls and the for-loop itself.
            i += inserted + 1
        else:
            i += 1
    return body


def hoist_loop_invariant_recips(body: list[ast.stmt]) -> list[ast.stmt]:
    """Apply the hoist pass to a list of kernel-body statements.

    Returns the (possibly modified) body.  Safe to call on any kernel body
    — only rewrites for-loops containing divisions where the divisor is a
    Name that is loop-invariant (transitively).

    Set ``HELION_DISABLE_HOIST_RECIP=1`` to skip the pass (escape hatch
    for cases where the extra reciprocal regresses register pressure).
    """
    import os

    if os.environ.get("HELION_DISABLE_HOIST_RECIP"):
        return body
    _HOIST_COUNTER[0] = 0
    return _hoist_in_body(body)
