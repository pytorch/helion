"""AST peepholes that:

1. Hoist ``x / scalar`` divisions out of inner loops when the divisor
   is loop-invariant (transitively).
2. Inline pure SSA-style ``NAME = ANOTHER_NAME`` alias chains where
   ``NAME`` is loop-invariant — letting the downstream code reference
   the loop-external root directly and dropping the redundant alias
   assignments.  This eliminates the per-iter "copy" that Helion's
   ``ast_extension`` passes routinely insert for SSA maintenance.

The CuTe softmax two-pass kernel computes ``out[k] = exp(x[k] - mi) / di``
in its second pass.  ``di`` is a single fp32 scalar per row that doesn't
change across the inner tile loop, but each inner iteration emits a
``v_12 = v_11 / di_copy_1_0`` floating-point divide.  Divides are slow
(~22 cycles on B200 vs 2 cycles for a multiply), and softmax fp16/bf16
shapes (e.g. (4096, 12672)) end up doing ~12672 divides per row per CTA.

The divide-hoist pass detects divisions inside for-loops where the
DIVISOR is loop-invariant — even when the divisor is a chain of copy
assignments (``di_copy_1 = di; di_copy_1_0 = di_copy_1``).  After
hoisting it walks outer-in so the rewrite places the reciprocal at the
OUTERMOST legal scope, avoiding cascade aliases like
``_helion_inv_div_0 = 1.0 * _helion_inv_div_1``.

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

The alias-DCE pass inlines chains like ``mi_copy_1 = mi`` /
``mi_copy_1_0 = mi_copy_1`` so the deepest inner-loop use can read
``mi`` directly.  This removes the per-iter "copy" instruction the SSA
maintenance otherwise leaves behind.

Measured impact on (4096, 12672) fp16 softmax_two_pass on B200:
  Before P16: 140us / 1486 GB/s
  After P16:  117us / 1776 GB/s   (+20% bench gain)
  After P17:  target 1800-1900 GB/s
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


def _is_placeholder_body(body: list[ast.stmt]) -> bool:
    """A body that contains an ``ast.Pass`` was intentionally emptied by
    an upstream pass (e.g. the tcgen05 role-lifecycle K-loop cleanup that
    replaces an owned slice with ``pass``).  Skip LICM/DCE inside such
    bodies so the explicit placeholder structure — including any
    snapshot aliases the test harness or downstream tooling inspects —
    survives intact.
    """
    return any(isinstance(s, ast.Pass) for s in body)


_RENAME_GROUPS: dict[str, str] = {}
# When True, the invariance analysis canonicalizes names through the
# rename group map; when False, names are used as-is.  The hoist passes
# that LIFT computation out of loops MUST canonicalize (so they don't
# treat ``mi`` and ``v_1_0`` as different vars), but the alias DCE pass
# only inlines within a single iter body — it does NOT need
# canonicalization, and using it would prevent the chain
# ``mi_copy_0 -> mi`` from resolving when ``mi`` will be re-assigned by
# a renamed alias later in the iter.
_USE_CANONICAL_INVARIANCE: list[bool] = [False]


def _canonical(name: str) -> str:
    """Map ``name`` to its post-rename canonical form, if a rename
    will collapse it AND canonical-aware mode is enabled.  Otherwise
    returns ``name`` unchanged.

    The renamer maps each name to the FIRST in its group; the canonical
    form is that target.  Two names that share a target collapse to the
    same canonical, and the invariance analysis must treat them as the
    SAME variable (for hoist-out passes).
    """
    if not _USE_CANONICAL_INVARIANCE[0]:
        return name
    return _RENAME_GROUPS.get(name, name)


def _collect_assigns_in_body(
    body: list[ast.stmt],
) -> tuple[dict[str, list[ast.expr]], set[str]]:
    """Walk ``body`` recursively and collect:
      - assigns_by_target: dict[name -> list of RHS expressions assigned]
      - all_written: set of all names that appear as Assign target anywhere
                     (including augmented assigns and tuple targets,
                     conservatively).

    Names are canonicalized via the rename group map when canonical-aware
    invariance mode is enabled.

    Returns both so callers can do transitive invariance analysis.
    """
    assigns_by_target: dict[str, list[ast.expr]] = {}
    all_written: set[str] = set()

    def _walk(stmts: list[ast.stmt]) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        canon = _canonical(target.id)
                        all_written.add(canon)
                        assigns_by_target.setdefault(canon, []).append(stmt.value)
                    elif isinstance(target, (ast.Tuple, ast.List)):
                        # Conservatively mark as non-invariant: we can't
                        # easily reason about destructuring.
                        for elt in target.elts:
                            if isinstance(elt, ast.Name):
                                all_written.add(_canonical(elt.id))
                                # Don't add to assigns_by_target -> treated
                                # as non-invariant.
            elif isinstance(stmt, ast.AugAssign):
                target = stmt.target
                if isinstance(target, ast.Name):
                    all_written.add(_canonical(target.id))
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

    Names are CANONICALIZED via the rename group map so that ``v_1_0``
    (which renames to ``mi``) is treated as the same variable as ``mi``.

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
    canon = _canonical(name)
    canon_target = _canonical(loop_target) if loop_target is not None else None
    if canon == canon_target:
        return False
    if canon in invariant_cache:
        return invariant_cache[canon]
    if canon in visiting:
        # Cycle — conservatively non-invariant.
        invariant_cache[canon] = False
        return False
    if canon not in all_written:
        # Defined outside the loop, not mutated inside — invariant.
        invariant_cache[canon] = True
        return True
    rhs_list = assigns_by_target.get(canon)
    if rhs_list is None:
        # Was written via tuple destructure or augassign — conservatively
        # non-invariant.
        invariant_cache[canon] = False
        return False

    visiting.add(canon)
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
    visiting.discard(canon)
    invariant_cache[canon] = invariant
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
    canon = _canonical(name)
    if canon in visited:
        return None
    if canon not in all_written:
        return canon
    rhs_list = assigns_by_target.get(canon)
    if rhs_list is None or len(rhs_list) != 1:
        return None
    rhs = rhs_list[0]
    if not isinstance(rhs, ast.Name):
        return None
    visited.add(canon)
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
    hoists fire.  Walks OUTER-IN so the outermost legal scope receives
    the hoist (avoids cascade aliases of the form
    ``_helion_inv_div_N = 1.0 * _helion_inv_div_{N+1}``).
    """
    if _is_placeholder_body(body):
        return body
    # OUTER-IN walk: hoist at this level first, then recurse into the
    # (possibly modified) child bodies.  By hoisting outermost first, the
    # nested bodies see ``... * _helion_inv_div_N`` (a Mult, not a Div)
    # and have nothing left to hoist — so they don't emit cascade aliases.
    i = 0
    while i < len(body):
        stmt = body[i]
        if isinstance(stmt, ast.For) and not stmt.orelse:
            inserted = _hoist_invariant_recips_in_for(stmt, body, i)
            # Skip past the inserted decls and the for-loop itself.
            i += inserted + 1
        else:
            i += 1

    # Now recurse into nested bodies for any remaining hoists that are
    # only legal at deeper scopes (e.g., divisor defined inside the outer
    # loop but invariant w.r.t. an inner loop).
    for stmt in body:
        if isinstance(stmt, (ast.For, ast.If, ast.With, ast.FunctionDef)):
            stmt.body = _hoist_in_body(stmt.body)
        if isinstance(stmt, (ast.For, ast.If)):
            stmt.orelse = _hoist_in_body(stmt.orelse)
    return body


class _NameReplacer(ast.NodeTransformer):
    """Replace every Load reference to ``old_name`` with ``new_name``.

    Does NOT rewrite Store / Del contexts (we only care about read sites).
    """

    def __init__(self, old_name: str, new_name: str) -> None:
        self.old_name = old_name
        self.new_name = new_name

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if isinstance(node.ctx, ast.Load) and node.id == self.old_name:
            return ast.copy_location(ast.Name(id=self.new_name, ctx=ast.Load()), node)
        return node


def _inline_invariant_aliases_in_body(body: list[ast.stmt]) -> None:
    """Inline pure SSA-style alias chains of the form ``NAME = ROOT_NAME``
    in ``body`` (single straight-line scope only).

    The alias chain ``mi_copy = mi; mi_copy_0 = mi_copy`` (typical of
    Helion's SSA maintenance, inserted by
    ``inductor_lowering.py::call_kernel_subgraph`` to handle phi nodes
    on loop inputs) collapses to direct ``mi`` reads with both alias
    assignments removed.

    Safety rule: an alias ``NAME = ROOT`` is inlinable iff
      1. ``NAME`` is assigned exactly ONCE in this body
      2. ``NAME`` has no Load references inside nested scopes (the
         ROOT's value seen there could differ from the snapshot)
      3. Every Load of ``NAME`` at this level occurs strictly BEFORE
         any assignment to ``ROOT`` at this level that comes after the
         alias definition (so the inlined ``ROOT`` read returns the
         same value the snapshot would have)

    Chains are resolved transitively: ``c_N -> c_{N-1} -> ... -> ROOT``
    inline directly to ``ROOT`` when every intermediate link satisfies
    the rule.

    Mutates ``body`` in place.  Does NOT recurse into nested bodies —
    the caller's outer walk handles those.
    """

    # Pass 1: collect single-Name aliases (target = Name) along with
    # position; remember when a name is assigned more than once so we
    # can disqualify it later.  Only the ``_copy`` snapshots that
    # ``inductor_lowering.call_kernel_subgraph`` inserts for loop-input
    # phi handling are eligible aliases — user-level reassignments
    # like ``mi = mi_next`` happen to share the ``target = Name`` shape
    # but are NOT snapshots and must be left intact (inlining them
    # would shift the loop-carry to the per-iter update value).
    def _is_helion_snapshot_alias(target: str, rhs: str) -> bool:
        # Helion's loop-input phi snapshot generates ``X_copy``,
        # ``X_copy_0``, ``X_copy_1_0`` etc. — all contain ``_copy`` and
        # have the chain root (or an earlier link of the same chain) as
        # the RHS Name.  User-level reassignments (``mi = mi_next``)
        # don't contain ``_copy`` in the target and stay intact.
        return "_copy" in target and target.startswith(rhs)

    alias_def_idx: dict[str, int] = {}
    alias_rhs: dict[str, str] = {}
    multi_assigned: set[str] = set()
    non_alias_assign_idx: dict[str, list[int]] = {}
    for idx, stmt in enumerate(body):
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        if not isinstance(target, ast.Name):
            continue
        name = target.id
        # A snapshot-shaped name (``X_copy_..._N``) is load-bearing
        # when it participates in an ``ast_rename`` group — either as
        # a variant that collapses to a user-visible canonical, or as
        # the canonical itself when other variants rename TO it.  In
        # either case the assignment is the end-of-iter loop-carry
        # update for some user variable; treat it as a non-alias
        # assignment so the rewrite still inlines its RHS alias chain
        # but the statement itself is preserved.
        is_loop_carry_snapshot = name in _RENAME_GROUPS
        if (
            isinstance(stmt.value, ast.Name)
            and not is_loop_carry_snapshot
            and _is_helion_snapshot_alias(name, stmt.value.id)
        ):
            if name in alias_def_idx or name in non_alias_assign_idx:
                multi_assigned.add(name)
            alias_def_idx[name] = idx
            alias_rhs[name] = stmt.value.id
        else:
            if name in alias_def_idx:
                multi_assigned.add(name)
            non_alias_assign_idx.setdefault(name, []).append(idx)
    for name in multi_assigned:
        alias_def_idx.pop(name, None)
        alias_rhs.pop(name, None)

    if not alias_def_idx:
        return

    # Pass 2: resolve each alias to its chain root (following
    # alias_rhs until we hit a non-alias name).
    def _resolve(name: str, visited: set[str]) -> str | None:
        if name in visited:
            return None  # cycle
        if name not in alias_rhs:
            return name
        visited.add(name)
        return _resolve(alias_rhs[name], visited)

    alias_to_root: dict[str, str] = {}
    for name in alias_def_idx:
        root = _resolve(name, set())
        if root is not None and root != name:
            alias_to_root[name] = root

    if not alias_to_root:
        return

    # Pass 3: collect Load refs per name at THIS level and per name
    # inside nested scopes.  We need to bail when a name is referenced
    # inside a nested for/if/with body, because the ROOT's value at
    # that point may differ from the alias snapshot.
    top_level_loads: dict[str, list[int]] = {n: [] for n in alias_to_root}
    nested_loads: set[str] = set()

    def _collect_loads(node: ast.AST, into: set[str]) -> None:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                into.add(sub.id)

    for idx, stmt in enumerate(body):
        # Skip the alias's own RHS (a single Name that's about to be dropped).
        is_alias_def = (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id in alias_to_root
            and isinstance(stmt.value, ast.Name)
        )
        if is_alias_def:
            continue
        # Top-level loads (RHS / target subscripts of straight-line stmts).
        if isinstance(stmt, ast.Assign):
            loads_here: set[str] = set()
            _collect_loads(stmt.value, loads_here)
            for target in stmt.targets:
                if not isinstance(target, ast.Name):
                    _collect_loads(target, loads_here)
            for name in loads_here:
                if name in top_level_loads:
                    top_level_loads[name].append(idx)
        elif isinstance(stmt, ast.AugAssign):
            loads_here = set()
            _collect_loads(stmt.value, loads_here)
            _collect_loads(stmt.target, loads_here)
            for name in loads_here:
                if name in top_level_loads:
                    top_level_loads[name].append(idx)
        elif isinstance(stmt, ast.Expr):
            loads_here = set()
            _collect_loads(stmt.value, loads_here)
            for name in loads_here:
                if name in top_level_loads:
                    top_level_loads[name].append(idx)
        elif isinstance(stmt, (ast.For, ast.If, ast.While, ast.With)):
            # Iter / test / context expressions are top-level reads.
            header_loads: set[str] = set()
            if isinstance(stmt, ast.For):
                _collect_loads(stmt.iter, header_loads)
            elif isinstance(stmt, (ast.If, ast.While)):
                _collect_loads(stmt.test, header_loads)
            for name in header_loads:
                if name in top_level_loads:
                    top_level_loads[name].append(idx)
            # Body / orelse loads are nested.
            for nested_stmt in getattr(stmt, "body", []):
                _collect_loads(nested_stmt, nested_loads)
            for nested_stmt in getattr(stmt, "orelse", []) or []:
                _collect_loads(nested_stmt, nested_loads)

    # Drop aliases used inside nested scopes — the snapshot semantics
    # don't survive the boundary.
    for name in list(alias_to_root):
        if name in nested_loads:
            del alias_to_root[name]
            top_level_loads.pop(name, None)

    if not alias_to_root:
        return

    # Pass 4: positional safety check.  The chain's snapshot is taken at
    # the FIRST link (the one whose RHS is the chain root); inlining a
    # leaf use to the root is safe only when the root is not reassigned
    # between that snapshot position and every use of the leaf.

    def _snapshot_idx(alias: str) -> int:
        """Walk the chain to the link whose RHS is the chain root and
        return its definition index — that's when the snapshot was
        taken.  For a single-link alias this is just ``alias_def_idx[alias]``.
        """
        current = alias
        seen: set[str] = set()
        while current in alias_rhs and alias_rhs[current] in alias_rhs:
            if current in seen:
                return alias_def_idx[alias]
            seen.add(current)
            current = alias_rhs[current]
        return alias_def_idx.get(current, alias_def_idx[alias])

    safe_aliases: dict[str, str] = {}
    for alias, root in alias_to_root.items():
        snapshot_idx = _snapshot_idx(alias)
        uses = top_level_loads.get(alias, [])
        # ROOT reassignments at this level: any non-alias assignment
        # of ROOT, plus any alias assignment to ROOT (those candidates
        # are dropped only after the safety check, so they still count
        # as reassignment points if they exist).
        root_assign_idxs = list(non_alias_assign_idx.get(root, []))
        if root in alias_def_idx:
            root_assign_idxs.append(alias_def_idx[root])
        # Reassignments strictly between the snapshot and any use are
        # unsafe to inline through.
        if not uses:
            # Unused alias; safe to drop (no replacements needed).
            safe_aliases[alias] = root
            continue
        last_use = max(uses)
        offending = [i for i in root_assign_idxs if snapshot_idx <= i <= last_use]
        if not offending:
            safe_aliases[alias] = root

    if not safe_aliases:
        return

    # Pass 5: rewrite the body in place — drop the dead alias defs and
    # rewrite the Load references at this level to point at the chain root.
    new_stmts: list[ast.stmt] = []
    for stmt in body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id in safe_aliases
            and isinstance(stmt.value, ast.Name)
        ):
            continue
        for alias, root in safe_aliases.items():
            stmt = _NameReplacer(alias, root).visit(stmt)
            ast.fix_missing_locations(stmt)
        new_stmts.append(stmt)
    body[:] = new_stmts


def _inline_invariant_aliases(body: list[ast.stmt]) -> list[ast.stmt]:
    """Walk down into every nested loop body and inline alias chains
    whose chain root is loop-external for THAT body.

    By doing this at every scope, even nested aliases get cleaned up.
    """
    if _is_placeholder_body(body):
        return body
    # Process this body first.
    _inline_invariant_aliases_in_body(body)
    # Recurse into for/if/with bodies that remain after rewrite.
    for stmt in body:
        if isinstance(stmt, (ast.For, ast.If, ast.With, ast.FunctionDef)):
            _inline_invariant_aliases(stmt.body)
        if isinstance(stmt, (ast.For, ast.If)):
            _inline_invariant_aliases(stmt.orelse)
    return body


_SCALE_COUNTER: list[int] = [0]


def _new_scaled_name() -> str:
    name = f"_helion_scaled_{_SCALE_COUNTER[0]}"
    _SCALE_COUNTER[0] += 1
    return name


def _is_numeric_const(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float))


def _expr_text(node: ast.expr) -> str:
    """ast.unparse the expression text (used to canonicalize hoist keys)."""
    return ast.unparse(node)


def _find_sub_assign_in_body(
    name: str,
    assigns_by_target: dict[str, list[ast.expr]],
) -> ast.BinOp | None:
    """If ``name`` has exactly ONE assignment whose RHS is a ``BinOp(Sub)``
    of the form ``A - INV`` (where INV is a Name), return that Sub.

    Otherwise return None.
    """
    rhs_list = assigns_by_target.get(name)
    if rhs_list is None or len(rhs_list) != 1:
        return None
    rhs = rhs_list[0]
    if isinstance(rhs, ast.BinOp) and isinstance(rhs.op, ast.Sub):
        return rhs
    return None


def _find_invariant_scale_subs(
    loop_body: list[ast.stmt],
    loop_target: str | None,
) -> dict[tuple[str, float], list[tuple[ast.BinOp, ast.BinOp, str]]]:
    """Walk loop_body and find one of two patterns equivalent to
    ``(A - B) * CONST`` where one of A/B is loop-invariant:

      Pattern A (in-place Sub):
        ``BinOp(Mult)`` with a numeric Constant on one side and the
        other side either a ``BinOp(Sub)`` directly or a single-arg Call
        (cast) wrapping a Sub.

      Pattern B (Sub assigned to a Name earlier):
        ``BinOp(Mult)`` with a numeric Constant on one side and the
        other side a Name (possibly cast-wrapped) where the Name has
        exactly ONE Assign in the loop body whose RHS is a Sub.

    For each match, returns a tuple ``(mult_binop, sub_binop, inv_side)``
    where ``inv_side`` is ``"right"`` (``A - INV``) or ``"left"``
    (``INV - A``).

    Maps ``(root_name_of_INVARIANT, const_value)`` -> list of those
    tuples so the caller can rewrite each to use a single hoisted
    ``INVARIANT * CONST`` scaled name.
    """
    assigns_by_target, all_written = _collect_assigns_in_body(loop_body)
    invariant_cache: dict[str, bool] = {}
    found: dict[tuple[str, float], list[tuple[ast.BinOp, ast.BinOp, str]]] = {}

    def _peel_cast(node: ast.expr) -> ast.expr:
        """If ``node`` is a Call(func, [arg], []), return ``arg``; else node."""
        if isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords:
            return node.args[0]
        return node

    for stmt in loop_body:
        for sub in ast.walk(stmt):
            if not (isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.Mult)):
                continue
            # Identify which side is the constant.
            const_side: ast.Constant | None = None
            other_side: ast.expr | None = None
            if _is_numeric_const(sub.right):
                const_side = sub.right  # type: ignore[assignment]
                other_side = sub.left
            elif _is_numeric_const(sub.left):
                const_side = sub.left  # type: ignore[assignment]
                other_side = sub.right
            if const_side is None or other_side is None:
                continue
            # Peel off a single cast wrapper if present.
            inner = _peel_cast(other_side)

            sub_node: ast.BinOp | None = None
            # Pattern A: inner is a Sub directly.
            if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Sub):
                sub_node = inner
            # Pattern B: inner is a Name that points to a single-assigned Sub.
            elif isinstance(inner, ast.Name):
                sub_node = _find_sub_assign_in_body(inner.id, assigns_by_target)

            if sub_node is None:
                continue
            # Try invariant on the RIGHT of the Sub first (``A - INV``);
            # fall back to LEFT (``INV - A``).  Both rewrites still
            # preserve sign order.
            inv_side: str  # 'right' or 'left'
            inv_name: str | None = None
            if isinstance(sub_node.right, ast.Name) and _is_loop_invariant(
                sub_node.right.id,
                assigns_by_target,
                all_written,
                loop_target,
                invariant_cache,
                set(),
            ):
                inv_side = "right"
                inv_name = sub_node.right.id
            elif isinstance(sub_node.left, ast.Name) and _is_loop_invariant(
                sub_node.left.id,
                assigns_by_target,
                all_written,
                loop_target,
                invariant_cache,
                set(),
            ):
                inv_side = "left"
                inv_name = sub_node.left.id
            else:
                continue
            assert inv_name is not None
            root = _resolve_root_name(inv_name, assigns_by_target, all_written, set())
            if root is None:
                continue
            # For Pattern B, we need the LHS of the Sub (``A``) to also
            # be a single-assigned variable so we can do the rewrite
            # safely without re-evaluating its RHS multiple times.
            # Pattern A doesn't need this since the Sub stays in place.
            const_val = const_side.value
            if not isinstance(const_val, (int, float)):
                continue
            key = (root, float(const_val))
            found.setdefault(key, []).append((sub, sub_node, inv_side))
    return found


def _rewrite_scale_sub_to_fma(
    node: ast.BinOp,
    sub_node: ast.BinOp,
    scaled_name: str,
    inv_side: str,
    assigns_by_target: dict[str, list[ast.expr]],
) -> None:
    """In-place rewrite of the outer ``cast(SUB) * CONST`` Mult.

    Two cases:

      ``inv_side == "right"``: ``cast(A - INV) * CONST`` becomes
      ``cast(A) * CONST - SCALED_NAME`` where ``SCALED_NAME = INV * CONST``.

      ``inv_side == "left"``: ``cast(INV - A) * CONST`` becomes
      ``SCALED_NAME - cast(A) * CONST``.

    The cast wrapper (if present) is preserved on the non-invariant
    side so the type stays consistent.  When ``A`` is a Name whose
    single assignment is already a ``cast(...)`` with the same target
    type, the outer cast is a redundant no-op and we strip it.
    """
    # Identify the const + other side again.
    if _is_numeric_const(node.right):
        other_side = node.left
        const_node = node.right
    else:
        other_side = node.right
        const_node = node.left

    # Peel off a single cast (Call(func, [arg], [])) wrapper.
    cast_call: ast.Call | None = None
    if (
        isinstance(other_side, ast.Call)
        and len(other_side.args) == 1
        and not other_side.keywords
    ):
        cast_call = other_side

    # The non-invariant operand of the Sub ("A") becomes the operand of
    # the new Mult.
    a_expr = sub_node.left if inv_side == "right" else sub_node.right

    # Build ``cast(A) * CONST`` (preserving cast if present), but skip
    # the outer cast when ``A`` is a Name already assigned ``cast(X)``
    # of the same target type — the outer wrap is a no-op.
    needs_cast = cast_call is not None
    if needs_cast and isinstance(a_expr, ast.Name) and cast_call is not None:
        a_rhs_list = assigns_by_target.get(a_expr.id)
        if a_rhs_list and len(a_rhs_list) == 1:
            a_rhs = a_rhs_list[0]
            if isinstance(a_rhs, ast.Call) and ast.unparse(a_rhs.func) == ast.unparse(
                cast_call.func
            ):
                needs_cast = False
    if needs_cast and cast_call is not None:
        new_a_mult_inner: ast.expr = ast.Call(
            func=cast_call.func, args=[a_expr], keywords=[]
        )
    else:
        new_a_mult_inner = a_expr
    new_a_mult = ast.BinOp(left=new_a_mult_inner, op=ast.Mult(), right=const_node)

    # Replace the outer Mult node with the FMA-friendly Sub.
    node.op = ast.Sub()
    if inv_side == "right":
        node.left = new_a_mult
        node.right = ast.Name(id=scaled_name, ctx=ast.Load())
    else:
        node.left = ast.Name(id=scaled_name, ctx=ast.Load())
        node.right = new_a_mult
    ast.fix_missing_locations(node)


def _hoist_invariant_scaled_subs_in_for(
    for_node: ast.For,
    parent_body: list[ast.stmt],
    for_idx: int,
) -> int:
    """Hoist ``INV * CONST`` out of ``(A - INV) * CONST`` patterns where
    INV is loop-invariant.

    Returns the count of statements inserted before ``for_idx``.
    """
    loop_target = for_node.target.id if isinstance(for_node.target, ast.Name) else None
    scale_map = _find_invariant_scale_subs(for_node.body, loop_target)
    if not scale_map:
        return 0

    # Re-collect assigns so the rewrite can drop redundant casts.
    assigns_by_target, _ = _collect_assigns_in_body(for_node.body)

    inserted = 0
    for (root_name, const_value), match_list in scale_map.items():
        scaled_name = _new_scaled_name()
        const_text = repr(const_value)
        decl = statement_from_string(f"{scaled_name} = {root_name} * {const_text}")
        parent_body.insert(for_idx + inserted, decl)
        inserted += 1
        for mult_node, sub_node, inv_side in match_list:
            _rewrite_scale_sub_to_fma(
                mult_node, sub_node, scaled_name, inv_side, assigns_by_target
            )
    return inserted


def _hoist_scaled_subs_in_body(body: list[ast.stmt]) -> list[ast.stmt]:
    """Outer-in walk to hoist ``(A - INV) * CONST`` patterns."""
    if _is_placeholder_body(body):
        return body
    i = 0
    while i < len(body):
        stmt = body[i]
        if isinstance(stmt, ast.For) and not stmt.orelse:
            inserted = _hoist_invariant_scaled_subs_in_for(stmt, body, i)
            i += inserted + 1
        else:
            i += 1

    for stmt in body:
        if isinstance(stmt, (ast.For, ast.If, ast.With, ast.FunctionDef)):
            stmt.body = _hoist_scaled_subs_in_body(stmt.body)
        if isinstance(stmt, (ast.For, ast.If)):
            stmt.orelse = _hoist_scaled_subs_in_body(stmt.orelse)
    return body


def _collect_all_name_reads(body: list[ast.stmt]) -> set[str]:
    """Walk ``body`` recursively and return the set of all Name LOAD
    references anywhere in any statement RHS, expression, condition, or
    iter.

    Used by the DCE pass to identify Assign statements whose target is
    never read afterwards.
    """
    reads: set[str] = set()

    def _walk(stmts: list[ast.stmt]) -> None:
        for stmt in stmts:
            # Skip the LHS targets of Assign but include the RHS.  Use
            # ast.walk on stmt.value to collect Loads.
            if isinstance(stmt, ast.Assign):
                reads.update(_names_read(stmt.value))
                # Also collect from any non-Name LHS components
                # (Subscripts, Attributes — those count as reads of
                # base names).
                for target in stmt.targets:
                    if not isinstance(target, ast.Name):
                        reads.update(_names_read(target))
                continue
            if isinstance(stmt, ast.AugAssign):
                reads.update(_names_read(stmt.value))
                reads.update(_names_read(stmt.target))
                continue
            if isinstance(stmt, ast.Expr):
                reads.update(_names_read(stmt.value))
                continue
            if isinstance(stmt, ast.If):
                reads.update(_names_read(stmt.test))
                _walk(stmt.body)
                _walk(stmt.orelse)
                continue
            if isinstance(stmt, ast.For):
                reads.update(_names_read(stmt.iter))
                _walk(stmt.body)
                _walk(stmt.orelse)
                continue
            if isinstance(stmt, ast.While):
                reads.update(_names_read(stmt.test))
                _walk(stmt.body)
                _walk(stmt.orelse)
                continue
            if isinstance(stmt, (ast.With, ast.FunctionDef)):
                _walk(stmt.body)
                continue
            if isinstance(stmt, ast.Return):
                if stmt.value is not None:
                    reads.update(_names_read(stmt.value))
                continue
            # Fallback: walk the whole stmt.
            reads.update(_names_read(stmt))

    _walk(body)
    return reads


def _dce_pure_assigns(body: list[ast.stmt]) -> list[ast.stmt]:
    """Remove ``NAME = pure-expr`` statements where NAME is never read
    afterwards, and ``pure-expr`` has no side effects.

    A pure-expr is conservatively any expression with no Call nodes.
    The pass iterates until fixed point so cascaded dead chains
    (``v_10 = v_9 - mi; v_X = v_10 + 1``) DCE together — if the second
    is removed, the first becomes dead too.

    Reads are collected across the ENTIRE body before any DCE so an
    assignment inside an ``if`` branch is never removed when the name
    is read in an enclosing scope (the previous per-scope read set
    would miss those outer reads and silently drop the assignment).

    Mutates ``body`` in place.
    """

    def _is_pure(node: ast.expr) -> bool:
        return all(not isinstance(sub, ast.Call) for sub in ast.walk(node))

    def _eligible_target(stmt: ast.Assign) -> str | None:
        if len(stmt.targets) != 1:
            return None
        target = stmt.targets[0]
        if not isinstance(target, ast.Name):
            return None
        # Only DCE names that look like compiler-generated temporaries
        # (start with ``v_``, ``_helion_``, or single-letter + digits).
        # This conservatively avoids DCE'ing user-defined names that
        # may be load-bearing for the renamer.
        name = target.id
        if not name.startswith(
            ("v_", "_helion_", "_tile_unroll_", "_mask_to", "_fuse_cache_")
        ):
            return None
        # Skip names that will be renamed by ``ast_rename`` to a name
        # other than themselves (e.g. ``v_8 -> di``).  Such names ARE
        # load-bearing: removing them would drop the assignment to the
        # user-visible ``di`` post-rename.
        canon = _RENAME_GROUPS.get(name, name)
        if canon != name:
            return None
        return name

    def _walk(stmts: list[ast.stmt], outer_reads: set[str]) -> list[ast.stmt]:
        if _is_placeholder_body(stmts):
            return stmts
        # Recurse into children first so DCE at the leaves can propagate
        # upward via fixed-point iteration.  ``outer_reads`` carries reads
        # from enclosing scopes so a name assigned in an ``if`` branch is
        # preserved if it is read after the if/else.
        for stmt in stmts:
            if isinstance(stmt, (ast.For, ast.If, ast.With, ast.FunctionDef)):
                stmt.body = _walk(stmt.body, outer_reads)
            if isinstance(stmt, (ast.For, ast.If)):
                stmt.orelse = _walk(stmt.orelse, outer_reads)

        # Iterate to fixed point.
        while True:
            reads = _collect_all_name_reads(stmts) | outer_reads
            new_stmts = []
            removed = False
            for stmt in stmts:
                if isinstance(stmt, ast.Assign):
                    name = _eligible_target(stmt)
                    if name is not None and name not in reads and _is_pure(stmt.value):
                        removed = True
                        continue
                new_stmts.append(stmt)
            stmts = new_stmts
            if not removed:
                break
        return stmts

    # Seed with the full body's reads so the inner ``_walk`` calls see
    # every read across the entire kernel, not just reads local to their
    # own scope.
    body[:] = _walk(body, _collect_all_name_reads(body))
    return body


def hoist_loop_invariant_recips(
    body: list[ast.stmt],
    rename_groups: dict[str, str] | None = None,
) -> list[ast.stmt]:
    """Apply the hoist passes to a list of kernel-body statements.

    Runs three sub-passes:

      1. Inline pure SSA alias chains (``mi_copy_1 = mi`` →
         direct use of ``mi``) so subsequent passes see clean root names.
      2. Hoist ``x / scalar`` divisions where the divisor is
         loop-invariant (transitively) into ``inv = 1.0 / scalar +
         x * inv``.
      3. Hoist ``(A - INV) * CONST`` patterns where INV is
         loop-invariant — emits ``INV_scaled = INV * CONST`` outside
         and rewrites the inner expression to ``A * CONST - INV_scaled``
         (an FMA-friendly form).

    ``rename_groups`` (if provided) maps pre-rename name -> canonical
    post-rename name.  This lets the invariance analysis treat aliases
    that will collapse under ``ast_rename`` (e.g. ``v_1_0`` -> ``mi``)
    as the SAME variable.  Without it, the FMA hoist would mistakenly
    classify ``mi`` as loop-invariant in the reduce loop and capture
    its stale initial value.

    Returns the (possibly modified) body.  Safe to call on any kernel
    body — only rewrites loop bodies where the analysis can prove
    invariance.

    Set ``HELION_DISABLE_HOIST_RECIP=1`` to skip the entire pass (escape
    hatch for cases where the extra hoist regresses register pressure).
    """
    import os

    if os.environ.get("HELION_DISABLE_HOIST_RECIP"):
        return body
    _HOIST_COUNTER[0] = 0
    _SCALE_COUNTER[0] = 0
    # Stash the rename map for the duration of this pass.  Using a
    # module-level global keeps the helper signatures stable across
    # the recursive walk.
    global _RENAME_GROUPS
    prev_renames = _RENAME_GROUPS
    _RENAME_GROUPS = rename_groups or {}
    try:
        # 1. Alias DCE first — inlines pure ``NAME = ROOT`` chains
        #    (Helion's SSA snapshots) so subsequent passes see clean
        #    root-name views.  Runs WITHOUT canonical-aware mode: the
        #    inlining is purely local (read-before-write inside one
        #    iter body), and canonicalization would prevent
        #    perfectly-safe chains from resolving when the chain's
        #    root has a renamed alias.  Safety check inside
        #    ``_inline_invariant_aliases_in_body`` rejects any alias
        #    whose use crosses a root reassignment or escapes into a
        #    nested scope, preserving load-bearing loop-carry updates.
        _USE_CANONICAL_INVARIANCE[0] = False
        _inline_invariant_aliases(body)
        # 2. Hoist reciprocals — runs WITH canonical-aware mode so that
        #    ``di`` is not classified as invariant when ``v_8`` (renamed
        #    to ``di``) is reassigned inside the loop.
        _USE_CANONICAL_INVARIANCE[0] = True
        body = _hoist_in_body(body)
        # 3. Hoist scaled-sub patterns — same canonical-aware mode.
        body = _hoist_scaled_subs_in_body(body)
        # 4. DCE dead pure assigns left behind by the rewrites
        #    (e.g. ``v_10 = v_9 - mi`` now unused because the Mult was
        #    rewritten to read ``v_9`` directly).
        body = _dce_pure_assigns(body)
    finally:
        _USE_CANONICAL_INVARIANCE[0] = False
        _RENAME_GROUPS = prev_renames
    return body
