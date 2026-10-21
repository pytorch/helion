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
  - It is read but never written inside the loop body (nested loops
    included), OR
  - It is bound exactly ONCE inside the loop, by a plain ``name = expr``
    assignment reached only through nested ``for`` bodies, every read of
    the name inside the loop comes after that assignment in the
    assignment's own statement list (nested reads included), and the RHS
    is itself loop-invariant (transitively).  The write then runs before
    each read on every path of one iteration and no read can see the
    previous iteration's value, so the name is a per-iteration copy of an
    invariant.  ``d = e`` at the END of a loop that divides by ``d`` is a
    loop-carried rebinding, not an invariant: the first iteration divides
    by the value ``d`` had before the loop.  Anything else (several
    bindings, augmented or destructuring assignments, a binding inside an
    ``if``/``while``/``with``, a read before the write) is non-invariant.

The rewrite then hoists the reciprocal computation:

    INV_DIVISOR = 1.0 / DIVISOR         # hoisted above the loop
    for OFFSET in range(...):
        ...
        QUOT = NUMERATOR * INV_DIVISOR
        ...

Multiple divisions sharing the same loop-invariant divisor share a
single reciprocal computation.

3. Share one reciprocal between ``n >= 2`` divisions by the same divisor
   VALUE inside a single straight-line block, even when the divisor is
   NOT loop-invariant (so the hoist above cannot fire).  ``swiglu_bwd``
   computes, per element and inside the unrolled vector-lane loop::

       v_9 = v_7 + v_8
       v_10 = v_7 / v_9
       v_12 = v_11 / v_9
       v_13 = v_12 / v_9

   and the scalar ``truediv`` lowering emits the full IEEE divide
   sequence (~20 SASS instructions with a BSSY/BSYNC branch) for every
   quotient.  The block-local pass rewrites this to one IEEE reciprocal
   plus one FMUL per quotient::

       v_9 = v_7 + v_8
       _helion_inv_div_0 = 1.0 / v_9
       v_10 = v_7 * _helion_inv_div_0
       v_12 = v_11 * _helion_inv_div_0
       v_13 = v_12 * _helion_inv_div_0

   Divisions are grouped by (name, SSA version): every statement that
   may rebind the divisor (including nested bodies, conservatively)
   starts a new version, so ``a / d; d = d + 1; b / d`` is left alone.
   A lone division is never rewritten (a reciprocal costs the same as
   the division it would replace).

Numerics of the two reciprocal rewrites (2 and 3) are identical: every
quotient becomes ``fl(x * fl(1 / d))`` instead of the correctly rounded
``fl(x / d)``.  With the fp32 unit roundoff ``u = 2**-24`` the relative
error versus the exact quotient is at most ``2u + u**2`` (two roundings)
instead of ``u``, i.e. below 1.5 ulp of the exact quotient, and the
result is the correctly rounded quotient or one of its two fp32
neighbours (at most 1 ulp away from the IEEE division result).  The bound
assumes ``1 / d`` is a normal fp32 value (roughly ``2**-128 < |d| <=
2**126``); outside that range the reciprocal can overflow or lose bits
where the division would not, exactly as for the loop-invariant hoist.
Both rewrites use only IEEE-rounded ``/`` and ``*``, but they still change
results, and only where they fire: inside a loop for the hoist, and where
a block repeats a divisor for the sharing.  Which loops and blocks exist
depends on the config (a lane loop when ``num_threads`` is below the block
size, a rolled reduction loop, an unrolled lane), so the rewrites run only
under the ``fast_math`` setting; configs must never change numerics.
``HELION_DISABLE_HOIST_RECIP`` turns off the whole pass.

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
import operator
from typing import TYPE_CHECKING
from typing import Literal
from typing import TypeVar
from typing import cast

import torch

from ..ast_extension import ExtendedAST
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ._ast_pass_utils import _names_read
from .scalar_recipe_rounding import _kind
from .scaled_sub_fusion import SCALED_SUBTRACTION_ATTR

if TYPE_CHECKING:
    from collections.abc import Collection
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import Mapping

_AstT = TypeVar("_AstT", bound=ast.AST)

_HOIST_COUNTER: list[int] = [0]


def _new_inv_name() -> str:
    name = f"_helion_inv_div_{_HOIST_COUNTER[0]}"
    _HOIST_COUNTER[0] += 1
    return name


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
    """Walk the loop body ``body`` recursively and collect:
      - all_written: every (canonical) name bound anywhere in it -- plain
        and augmented assignments, destructuring, nested loop targets,
        ``with`` items, walrus targets -- nested bodies included.
      - assigns_by_target: ``name -> [rhs]`` for the names whose ONE binding
        in the body is a plain ``name = rhs`` assignment that provably runs
        before every read of the name within one iteration: the assignment
        is reached only through nested ``for`` bodies (no ``if``/``while``/
        ``with``/``orelse`` between the loop and it) and every read of the
        name in the whole body sits after it in its own statement list
        (nested reads included).  A read before the write -- in an earlier
        statement, in the write's own RHS, or anywhere outside the write's
        statement list -- is a loop-carried use: it sees the previous
        iteration's (or the pre-loop) value, so the name is left out of
        ``assigns_by_target`` and ``_is_loop_invariant`` rejects it.

    Names are canonicalized via the rename group map when canonical-aware
    invariance mode is enabled, so ``v_9 = e`` with ``v_9`` renaming to
    ``d`` counts as a binding of ``d``.

    Returns both so callers can do transitive invariance analysis.
    """
    all_written: set[str] = set()
    # canonical name -> [(statement, its statement list, index, straight)]
    bindings: dict[str, list[tuple[ast.stmt, list[ast.stmt], int, bool]]] = {}

    def _record(name: str, entry: tuple[ast.stmt, list[ast.stmt], int, bool]) -> None:
        canon = _canonical(name)
        all_written.add(canon)
        bindings.setdefault(canon, []).append(entry)

    def _walk(stmts: list[ast.stmt], straight: bool) -> None:
        for idx, stmt in enumerate(stmts):
            entry = (stmt, stmts, idx, straight)
            if isinstance(stmt, (ast.For, ast.While, ast.If, ast.With, ast.Try)):
                # The header's own bindings (loop target, ``with`` items,
                # walrus targets in a test or iterator) belong to this
                # position; the nested bodies are walked below.  Only a
                # ``for`` body keeps the straight-line property.
                for child in ast.iter_child_nodes(stmt):
                    if isinstance(child, (ast.stmt, ast.excepthandler)):
                        continue
                    for name in _names_bound_by(child):  # pyrefly: ignore
                        _record(name, entry)
                for attr in ("body", "orelse", "finalbody"):
                    nested = getattr(stmt, attr, None) or []
                    _walk(
                        nested,
                        straight and isinstance(stmt, ast.For) and attr == "body",
                    )
                for handler in getattr(stmt, "handlers", None) or []:
                    _walk(handler.body, False)
            else:
                for name in _names_bound_by(stmt):
                    _record(name, entry)

    _walk(body, True)

    def _reads(node: ast.AST, canon: str) -> int:
        return sum(
            1
            for sub in ast.walk(node)
            if isinstance(sub, ast.Name)
            and isinstance(sub.ctx, ast.Load)
            and _canonical(sub.id) == canon
        )

    assigns_by_target: dict[str, list[ast.expr]] = {}
    for canon, entries in bindings.items():
        if len(entries) != 1:
            continue
        stmt, stmts, idx, straight = entries[0]
        if not (
            straight
            and isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and _names_bound_by(stmt) == {stmt.targets[0].id}
        ):
            continue
        # Every read of the name must follow the write in its own list.
        total_reads = sum(_reads(top, canon) for top in body)
        dominated_reads = sum(_reads(later, canon) for later in stmts[idx + 1 :])
        if total_reads != dominated_reads:
            continue
        assigns_by_target[canon] = [stmt.value]
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
      - A name with exactly one binding in the loop, a plain assignment
        that runs before every read of the name within one iteration
        (``_collect_assigns_in_body`` admits it to ``assigns_by_target``
        only then), whose RHS is invariant, is invariant.
      - All other names (several bindings, a read before the write such as
        ``d = e`` at the end of a loop dividing by ``d``, a binding inside
        an ``if``/``while``/``with``, augassigns, tuple-destructured,
        etc.) are non-invariant.

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

    Every link must be in ``assigns_by_target``, i.e. a plain assignment
    that runs before each of its reads within one iteration
    (``_collect_assigns_in_body``); a link written anywhere else in the
    loop, or read before it is written, ends the resolution.  Returns the
    outermost name (one not in ``all_written``) or None when the chain
    can't be cleanly resolved (multiple defs, non-Name RHS, cycle, ...).
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
    live_recips: dict[str, str] | None = None,
) -> int:
    """Try to hoist loop-invariant divisions in a single for-loop.

    ``live_recips`` maps divisor names to reciprocals hoisted earlier in the
    same body that are still valid (the divisor was not rebound since); a
    later sibling loop dividing by the same name reuses them instead of
    computing another reciprocal.  Returns the number of statements inserted
    before ``for_idx`` (so the caller can adjust subsequent indices).
    """
    loop_target = for_node.target.id if isinstance(for_node.target, ast.Name) else None
    divisor_map = _find_invariant_div_binops(for_node.body, loop_target)
    if not divisor_map:
        return 0

    inserted = 0
    for divisor_name, binop_list in divisor_map.items():
        inv_name = live_recips.get(divisor_name) if live_recips is not None else None
        if inv_name is None:
            inv_name = _new_inv_name()
            decl = statement_from_string(f"{inv_name} = 1.0 / {divisor_name}")
            parent_body.insert(for_idx + inserted, decl)
            inserted += 1
            if live_recips is not None:
                live_recips[divisor_name] = inv_name
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
    # Reciprocals hoisted at this level stay valid for later sibling loops
    # until any statement (at this level or nested) binds their divisor again.
    live_recips: dict[str, str] = {}
    i = 0
    while i < len(body):
        stmt = body[i]
        if isinstance(stmt, ast.For) and not stmt.orelse:
            inserted = _hoist_invariant_recips_in_for(stmt, body, i, live_recips)
            # Skip past the inserted decls and the for-loop itself.
            i += inserted + 1
        else:
            i += 1
        for bound in _names_bound_by(stmt):
            live_recips.pop(_canonical(bound), None)
            live_recips.pop(bound, None)

    # Now recurse into nested bodies for any remaining hoists that are
    # only legal at deeper scopes (e.g., divisor defined inside the outer
    # loop but invariant w.r.t. an inner loop).
    for stmt in body:
        if isinstance(stmt, (ast.For, ast.While, ast.If, ast.With, ast.FunctionDef)):
            stmt.body = _hoist_in_body(stmt.body)
        if isinstance(stmt, (ast.For, ast.While, ast.If)):
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
            # Also record this write under its canonical (post-rename)
            # name so the Pass-4 root-reassignment safety check can see
            # that a snapshot's chain root is reassigned through a renamed
            # alias (e.g. ``v_9 = acc_cnt + ...`` writes canonical
            # ``acc_cnt``). Without this, a snapshot taken before that
            # write could be inlined past the reassignment, dropping the
            # loop-carried value.
            canon = _RENAME_GROUPS.get(name, name)
            if canon != name:
                non_alias_assign_idx.setdefault(canon, []).append(idx)
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
        # Also count reassignments of ROOT's canonical (post-rename)
        # name: a renamed alias like ``v_9 = acc_cnt + ...`` writes the
        # same logical loop-carried variable as ``root``, and inlining
        # the snapshot past it would drop the carried value.
        canon_root = _RENAME_GROUPS.get(root, root)
        if canon_root != root:
            root_assign_idxs.extend(non_alias_assign_idx.get(canon_root, []))
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
    # Recurse into for/while/if/with bodies that remain after rewrite.
    for stmt in body:
        if isinstance(stmt, (ast.For, ast.While, ast.If, ast.With, ast.FunctionDef)):
            _inline_invariant_aliases(stmt.body)
        if isinstance(stmt, (ast.For, ast.While, ast.If)):
            _inline_invariant_aliases(stmt.orelse)
    return body


_SCALE_COUNTER: list[int] = [0]


def _new_scaled_name() -> str:
    name = f"_helion_scaled_{_SCALE_COUNTER[0]}"
    _SCALE_COUNTER[0] += 1
    return name


def _is_numeric_const(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float))


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
            # This reassociation targets floating-point FMA expressions. Without
            # a typed integer proof, retain integer arithmetic exactly: turning
            # its scale into a float can round indices and add conversions.
            if not isinstance(const_val, float):
                continue
            key = (root, const_val)
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
    setattr(node, SCALED_SUBTRACTION_ATTR, True)
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
        if isinstance(stmt, (ast.For, ast.While, ast.If, ast.With, ast.FunctionDef)):
            stmt.body = _hoist_scaled_subs_in_body(stmt.body)
        if isinstance(stmt, (ast.For, ast.While, ast.If)):
            stmt.orelse = _hoist_scaled_subs_in_body(stmt.orelse)
    return body


# Statements whose expressions are evaluated exactly once, in program
# order, when the statement runs.  Divisions inside compound statements
# (for/if/while headers) are left to the hoist above; their bodies are
# visited as blocks of their own.
_EAGER_STMT_TYPES = (
    ast.Assign,
    ast.AugAssign,
    ast.AnnAssign,
    ast.Expr,
    ast.Return,
    ast.Assert,
)
# Expressions whose body runs later (or repeatedly) in its own scope; a
# reciprocal computed before the enclosing statement may be stale there.
_DEFERRED_EXPR_TYPES = (
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


def _eager_nodes(stmt: ast.stmt) -> list[ast.AST]:
    """Breadth-first list of every node of ``stmt`` that is evaluated when
    ``stmt`` runs, skipping ``_DEFERRED_EXPR_TYPES`` subtrees."""
    nodes: list[ast.AST] = []
    todo: list[ast.AST] = [stmt]
    while todo:
        node = todo.pop(0)
        if isinstance(node, _DEFERRED_EXPR_TYPES):
            continue
        nodes.append(node)
        todo.extend(ast.iter_child_nodes(node))
    return nodes


def _names_bound_by(stmt: ast.stmt) -> set[str]:
    """Every name ``stmt`` may (re)bind anywhere inside it: ``Store``/``Del``
    ``Name`` contexts (assignment, for/with and walrus targets, ``del``)
    plus nested ``def`` / ``class`` / ``import`` bindings.  Nested bodies
    are included conservatively -- an ``if`` branch may or may not run,
    so any name it binds has to be treated as changed after the ``if``.
    """
    bound: set[str] = set()
    for node in ast.walk(stmt):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.alias):
            bound.add((node.asname or node.name).split(".")[0])
    return bound


def _is_reciprocal_div(node: ast.BinOp) -> bool:
    """``1.0 / d`` (or ``1 / d``) already IS a reciprocal -- e.g. a
    declaration the loop-invariant hoist emitted.  Rewriting it to
    ``1.0 * inv`` would only add a multiply."""
    left = node.left
    return (
        isinstance(left, ast.Constant)
        and isinstance(left.value, (int, float))
        and left.value == 1
    )


def _share_repeated_recips_in_block(body: list[ast.stmt]) -> None:
    """Sub-pass 3 for ONE straight-line block: divisions by the same
    divisor value share a single ``inv = 1.0 / d`` reciprocal.

    Divisions are grouped by ``(canonical divisor name, version)`` where
    the version is bumped after every statement that may rebind the
    divisor (``_names_bound_by``), so all members of a group observe the
    same value of ``d``.  Names are canonicalized through the rename-group
    map so a rebind through a to-be-renamed alias (``v_1_0`` -> ``mi``)
    also starts a new version.  Groups with at least two members get the
    reciprocal declared immediately before their first use (where ``d``
    is known to be bound) and every member rewritten to ``x * inv``.

    Mutates ``body`` in place.  Does NOT recurse into nested bodies --
    ``_share_repeated_recips_in_body`` handles those as separate blocks.
    """
    version: dict[str, int] = {}
    groups: dict[tuple[str, int], list[ast.BinOp]] = {}
    first_use: dict[tuple[str, int], int] = {}
    for idx, stmt in enumerate(body):
        if isinstance(stmt, _EAGER_STMT_TYPES):
            for node in _eager_nodes(stmt):
                if (
                    isinstance(node, ast.BinOp)
                    and isinstance(node.op, ast.Div)
                    and isinstance(node.right, ast.Name)
                    and not _is_reciprocal_div(node)
                ):
                    canon = _canonical(node.right.id)
                    key = (canon, version.get(canon, 0))
                    groups.setdefault(key, []).append(node)
                    first_use.setdefault(key, idx)
        # Rebinds take effect AFTER the statement's reads.
        for name in _names_bound_by(stmt):
            canon = _canonical(name)
            version[canon] = version.get(canon, 0) + 1

    decls: list[tuple[int, int, ast.stmt]] = []
    for key, binops in groups.items():
        if len(binops) < 2:
            continue
        inv_name = _new_inv_name()
        # Spell the divisor as the first member does: that name is read
        # by the very statement the declaration is placed in front of.
        divisor = binops[0].right
        assert isinstance(divisor, ast.Name)
        decl = statement_from_string(f"{inv_name} = 1.0 / {divisor.id}")
        decls.append((first_use[key], len(decls), decl))
        for binop in binops:
            _rewrite_div_to_mul(binop, inv_name)
    # Insert back-to-front so earlier indices stay valid; the descending
    # group order for same-index declarations keeps them in program order.
    for idx, _order, decl in sorted(decls, key=operator.itemgetter(0, 1), reverse=True):
        body.insert(idx, decl)


def _share_repeated_recips_in_body(body: list[ast.stmt]) -> list[ast.stmt]:
    """Apply ``_share_repeated_recips_in_block`` to ``body`` and, recursively,
    to every nested block (each is an independent straight-line scope)."""
    if _is_placeholder_body(body):
        return body
    _share_repeated_recips_in_block(body)
    for stmt in body:
        if isinstance(stmt, (ast.For, ast.While, ast.If, ast.With, ast.FunctionDef)):
            stmt.body = _share_repeated_recips_in_body(stmt.body)
        if isinstance(stmt, (ast.For, ast.While, ast.If)):
            stmt.orelse = _share_repeated_recips_in_body(stmt.orelse)
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
            if isinstance(
                stmt, (ast.For, ast.While, ast.If, ast.With, ast.FunctionDef)
            ):
                stmt.body = _walk(stmt.body, outer_reads)
            if isinstance(stmt, (ast.For, ast.While, ast.If)):
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


# --------------------------------------------------------------------------
# Exact loop-invariant fp32 division hoist (the IEEE result, bit for bit)
# --------------------------------------------------------------------------
#
# ``v = x / d`` with an fp32 scalar divisor ``d`` that is invariant across a
# ``for`` loop becomes, once per divisor above the loop::
#
#     _helion_exdiv_y_K = 1.0 / d                                   # RN(1/d)
#     _helion_exdiv_lo_K, _helion_exdiv_hi_K = _cute_exact_div_bounds_f32(d)
#
# and the loop is speculated (the "redo" form)::
#
#     _helion_exdiv_min_K = cutlass.Float32(float('inf'))   # running min |x|
#     _helion_exdiv_max_K = cutlass.Float32(0.0)            # running max |x|
#     _helion_exdiv_save_K_acc = acc                        # loop-carried values
#     for ...:                                              # the speculative pass
#         v = cutlass.Float32(_cute_exact_div_fast_f32(x, d, y))
#         _helion_exdiv_min_K = cute.arch.fmin(_helion_exdiv_min_K, cute.math.abs(x), nan=True)
#         _helion_exdiv_max_K = cute.arch.fmax(_helion_exdiv_max_K, cute.math.abs(x), nan=True)
#     if _cute_exact_div_redo_f32(_helion_exdiv_min_K, _helion_exdiv_max_K, lo, hi):
#         acc = _helion_exdiv_save_K_acc                    # roll back, then
#         for ...:                                          # the original loop
#             v = x / d
#
# i.e. the Markstein quotient ``q = RN(x*y); r = RN(x - d*q); RN(q + r*y)``
# per element plus two FMNMX, and the hard cases (some ``|x|`` outside the
# hoisted window) re-dividing with ``div.rn.f32`` itself.  Every dividend
# of the loop inside the window means every speculative quotient was the
# IEEE one and the pass computed exactly what the original loop computes;
# otherwise (a NaN dividend makes the maximum NaN and fails the test too)
# the original loop runs from the restored state and overwrites every
# store.  That is sound only for loops that read no memory they write
# (tensor arguments need a disjointness proof, register fragments are per
# thread), contain nothing but pure arithmetic, plain loads and stores and
# per-iteration Python lists (no barriers, shuffles, reductions, atomics,
# async copies, control flow or unknown helpers), take no quotient into an
# address, array index or trip count, and whose loop-carried values are DSL
# scalars; ``_redo_eligible`` checks each condition.  The redo form goes on
# the outermost loop that can be re-executed verbatim, else on the child
# loops that can; a site in no such loop keeps the plain ``x / d`` (and a
# divisor none of whose sites is placed is not hoisted at all: the render
# is the plain render).  A per-element guard (``if not admitted: x / d``,
# the retired ``guard`` form of ``cute_exact_div_form``) is NOT an option:
# its branch changes the compiled shape of the loop so much (no unrolling,
# spills, a BSSY/BRA/BSYNC trio per element) that ptxas contracts the plain
# lowering's unmarked fp32 ``mul``/``add``/``sub`` chains around the
# quotient differently, and reductions over the quotients differed from
# the plain render by 1-2 ulp on ragged shapes.
# The result equals ``x / d`` bit for bit --
# ``helion/_compiler/cute/exact_division.py`` states the argument (an
# exhaustive significand-pair check plus the exponent-scaling lemma) and
# ``test/test_cute_exact_division.py`` checks it -- so unlike the
# ``fast_math`` reciprocal hoist this one may depend on the loops a config
# creates.  Only the generic lowering's ``v_N = a / b`` statements are
# rewritten (hand-emitted flash/matmul bodies spell their temporaries
# differently and pin their own sequences), only when both operands are
# proved fp32 scalars (``cutlass.Float32(...)``, ``.bitcast(cutlass.Float32)``,
# ``cute.math`` results, fp32 tensor loads, fp32 scalar kernel arguments and
# arithmetic over them), never when a per-thread select (``... if mask
# else cutlass.Float32(0)``, the tail of a ragged reduction axis or a
# partial row block) may reach the dividend (``_select_derived_names``:
# the masked zero would make the redo fire for the tail threads of every
# row while the rest keep the speculative pass, and with both loop copies
# live ptxas contracted the fp32 chains around the quotient differently
# from the plain kernel -- 1 ulp in a sum of squares), and never for a
# reciprocal (``1.0 / d`` is already the hoisted form).  Integer ``//``/
# ``%`` and 16-bit divisions are untouched.
#
# Four rules keep the rewrite sound on the bodies codegen produces:
#
# * The divisor's loop invariance is the ``_collect_assigns_in_body`` proof:
#   the root and every alias link are unwritten in the loop (nested loops
#   included), or written once by a plain assignment that precedes every
#   read within one iteration.  ``for ...: v = x / d; d = e`` divides by the
#   pre-loop ``d`` first and by ``e`` afterwards; it is left alone.
# * Codegen emits a reduction sweep and its rematerialized store sweep with
#   the same statement objects; ``_dealias_shared_nodes`` gives every
#   position its own copy before the first in-place rewrite, so each sweep
#   is rewritten in its own context (its own hoisted names and divisor).
# * The values the redo copy rolls back are the names the loop rebinds that
#   were bound when it starts -- under the DSL's staging rules
#   (``_bound_before`` / ``_is_unrolled_region``): the kernel's parameters,
#   the statements before the loop, the headers of the loops enclosing it,
#   the bindings of every unrolled region (``cutlass.range_constexpr``,
#   ``cutlass.const_expr`` tests, ``with``) that ends before it, and of a
#   staged region (``range``, a dynamic test) only the names that existed
#   before that region.  Missing one is unsound (a stale accumulator after
#   the redo), so where the rule is uncertain the walk keeps the name.
# * The pass runs after every structural pass (``hoist_exact_invariant_divisions``
#   is called right before the rename), so the matchers that recognize the
#   plain ``v = x / d`` sweeps (the independent partial accumulators of
#   ``split_rolled_vector_sums``, pipelined/packed loads, the cluster pairs,
#   the scalar epilogue) see what the plain render sees and the redo render
#   keeps the plain render's structure and summation order.

EXACT_DIV_BOUNDS_HELPER = "_cute_exact_div_bounds_f32"
EXACT_DIV_PREFIX = "_helion_exdiv_"
_FP32_DTYPE_STR = "cutlass.Float32"
_EXACT_COUNTER: list[int] = [0]
_Kind = Literal["fp32", "literal"] | None


def _new_exact_names() -> tuple[str, str, str]:
    index = _EXACT_COUNTER[0]
    _EXACT_COUNTER[0] += 1
    return (
        f"{EXACT_DIV_PREFIX}y_{index}",
        f"{EXACT_DIV_PREFIX}lo_{index}",
        f"{EXACT_DIV_PREFIX}hi_{index}",
    )


# ``ExtendedAST.new`` carries these itself; every other non-field attribute
# of a node (the ``_helion_*`` markers passes leave on loops and
# statements) is copied by ``_deep_clone``.
_EXTENDED_AST_INTERNALS = frozenset(
    {"_location", "_type_info", "_loop_type", "_is_kernel_call", "_root_id"}
)


def _deep_clone(node: _AstT) -> _AstT:
    """A fresh copy of ``node``'s whole subtree that also keeps the marker
    attributes set on its nodes (``scalar_recipe._clone`` copies fields
    only), so a redo copy or a de-aliased statement is indistinguishable
    from the original to every later pass."""

    def clone_field(value: object) -> object:
        if isinstance(value, ast.AST):
            return _deep_clone(value)
        if isinstance(value, list):
            return [clone_field(item) for item in value]
        return value

    fields = {name: clone_field(value) for name, value in ast.iter_fields(node)}
    if isinstance(node, ExtendedAST):
        result = node.new(fields)
    else:
        result = ast.copy_location(type(node)(**fields), node)
    skip = set(node._fields) | set(node._attributes) | _EXTENDED_AST_INTERNALS
    for key, value in vars(node).items():
        if key not in skip and key not in vars(result):
            setattr(result, key, value)
    return cast("_AstT", result)


def _dealias_shared_nodes(body: list[ast.stmt]) -> None:
    """Give every position in the tree its own node object.

    Helion's codegen reuses statement objects: the rematerialized store
    sweep of a reduction is emitted with the SAME ``ast.Assign`` (and lane
    loop body) objects as the reduction sweep.  A rewrite of one occurrence
    in place is then a rewrite of both, with the other occurrence's context
    (its own hoisted names, its own divisor) ignored.  Every repeated node
    reached through a field or a body list is replaced by a deep clone, and
    a body list held by two nodes by its own list, before the pass rewrites
    anything; the rendered text is unchanged.
    """
    seen: set[int] = set()
    seen_lists: set[int] = set()

    def visit(node: ast.AST) -> None:
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                if id(value) in seen_lists:
                    value = list(value)
                    setattr(node, field, value)
                seen_lists.add(id(value))
                for index, item in enumerate(value):
                    if isinstance(item, ast.AST):
                        value[index] = fresh(item)
            elif isinstance(value, ast.AST):
                setattr(node, field, fresh(value))

    def fresh(node: ast.AST) -> ast.AST:
        if id(node) in seen:
            # The clone's subtree is new to the tree; no need to walk it.
            return _deep_clone(node)
        seen.add(id(node))
        visit(node)
        return node

    for index, stmt in enumerate(body):
        body[index] = cast("ast.stmt", fresh(stmt))


class _CanonicalKinds:
    """``name -> kind`` keyed by post-rename canonical names (so ``v_8`` and
    the ``di`` it renames to share one entry); ``_kind`` only calls ``get``.
    ``selected`` holds the names ``_select_derived_names`` found: a value
    that may come from a masked select is never a dividend of an exact
    division."""

    def __init__(self) -> None:
        self.kinds: dict[str, _Kind] = {}
        self.selected: set[str] = set()

    def get(self, name: str, default: _Kind = None) -> _Kind:
        return self.kinds.get(_canonical(name), default)

    def __contains__(self, name: str) -> bool:
        return _canonical(name) in self.kinds

    def select_derived(self, node: ast.expr) -> bool:
        """Whether ``node`` is, or reads as a value a name that may hold,
        the result of a per-thread select."""
        return _contains_select(node) or any(
            _canonical(name) in self.selected for name in _value_reads(node)
        )


# Cross-thread reductions: their result is one value for every thread of
# the group, so whatever a select did to their operands cannot make the
# redo decision differ between the threads of a row.
_REDUCTION_CALL_PREFIXES = (
    "_cute_grouped_reduce_",
    "_cute_lane_reduce_",
    "cute.arch.warp_reduction",
)


def _is_pointer_expr(node: ast.AST) -> bool:
    """``t.iterator + ...``: an address (the clamped operand of a packet
    load, ``ptr if lane < n else ptr0``, selects no value)."""
    if isinstance(node, ast.Attribute):
        return node.attr == "iterator"
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        return _is_pointer_expr(node.left) or _is_pointer_expr(node.right)
    if isinstance(node, ast.IfExp):
        return _is_pointer_expr(node.body) or _is_pointer_expr(node.orelse)
    return False


def _is_reduction_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and ast.unparse(node.func).startswith(
        _REDUCTION_CALL_PREFIXES
    )


def _value_walk(node: ast.AST) -> Iterator[ast.AST]:
    """``node``'s subtree minus addresses, subscript indices and the
    operands of cross-thread reductions: what flows into the VALUE of
    ``node`` thread by thread."""
    todo = [node]
    while todo:
        sub = todo.pop()
        if _is_pointer_expr(sub) or _is_reduction_call(sub):
            continue
        yield sub
        if isinstance(sub, ast.Subscript):
            todo.append(sub.value)
        else:
            todo.extend(ast.iter_child_nodes(sub))


def _contains_select(node: ast.AST) -> bool:
    """Whether a conditional expression selects a value of ``node``."""
    return any(isinstance(sub, ast.IfExp) for sub in _value_walk(node))


def _value_reads(node: ast.AST) -> set[str]:
    return {
        sub.id
        for sub in _value_walk(node)
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load)
    }


def _select_derived_names(body: list[ast.stmt]) -> set[str]:
    """Canonical names whose value may come from a per-thread conditional
    expression (``... if mask else cutlass.Float32(0)``: the packet form of
    a masked load or register-cache reload, the masked operand of a
    reduction) -- bound by one, or computed from such a name through
    arithmetic, a ``cutlass.Float32(...)`` / ``.to(...)`` conversion (which
    ``_kind`` accepts as fp32 whatever its operand) or a register cache
    written from one (``cache[k] = ...`` taints ``cache``).  Not through an
    address (``cute.arch.load(ptr if lane < n else ptr0, ...)`` clamps
    where the packet is read, not what it holds), a subscript index, or a
    cross-thread reduction (its result is row-uniform: the first-tile max
    and the NaN guard of the online softmax select, but every thread of
    the row then agrees).  Fixpoint over every assignment of the body.

    A division whose DIVIDEND is such a name keeps the plain division: a
    masked zero makes the redo fire for the tail threads of every row of a
    ragged reduction axis while the other threads keep the speculative
    pass, and the kernel with both loop copies live compiled the fp32
    chains around the quotient differently from the plain kernel (1 ulp in
    a sum of the quotients' squares, two rows of 53 x 4000).  The divisor
    is one value for every thread of the sweep (a row sum, a row scale),
    so a masked divisor fires the redo for all of them alike.  The scalar
    spelling of a masked load (``(t.iterator + i).load() if mask else 0``)
    already failed closed because ``_kind`` knows no ``.load()``; this is
    the same rule for every spelling."""
    assigns = [
        node
        for stmt in body
        for node in ast.walk(stmt)
        if isinstance(node, (ast.Assign, ast.AugAssign))
    ]
    facts = []
    for stmt in assigns:
        bound = {_canonical(name) for name in _names_bound_by(stmt)}
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        for target in targets:
            for sub in ast.walk(target):
                if isinstance(sub, ast.Subscript):
                    bound.update(_canonical(n) for n in _names_read(sub.value))
        reads = {_canonical(n) for n in _value_reads(stmt.value)}
        facts.append((_contains_select(stmt.value), reads, bound))
    tainted: set[str] = set()
    changed = True
    while changed:
        changed = False
        for selects, reads, bound in facts:
            if (selects or reads & tainted) and bound - tainted:
                tainted |= bound
                changed = True
    return tainted


def _iterator_tensor(node: ast.AST) -> str | None:
    """The tensor whose ``.iterator`` a pointer expression offsets."""
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Attribute)
            and sub.attr == "iterator"
            and isinstance(sub.value, ast.Name)
        ):
            return sub.value.id
    return None


def _expression_kind(
    node: ast.expr, kinds: _CanonicalKinds, tensor_dtypes: Mapping[str, str]
) -> _Kind:
    """``_kind`` plus the scalar loads: ``(t.iterator + ...).load()``, whose
    dtype is the tensor argument's, and the helper forms that spell it."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load"
        and not node.args
        and not node.keywords
    ):
        tensor = _iterator_tensor(node.func.value)
        if tensor is not None and tensor_dtypes.get(tensor) == _FP32_DTYPE_STR:
            return "fp32"
        return None
    if isinstance(node, ast.Call) and len(node.args) >= 2:
        # ``_cute_scalar_policy_load(ptr, cutlass.Float32, policy)`` and the
        # scalar ``cute.arch.load(ptr, cutlass.Float32, ...)`` spell their
        # dtype (the packet form passes an ``ir.VectorType`` instead).
        path = ast.unparse(node.func)
        if path in {"_cute_scalar_policy_load", "cute.arch.load"}:
            return "fp32" if ast.unparse(node.args[1]) == _FP32_DTYPE_STR else None
    return _kind(node, kinds)  # pyrefly: ignore[bad-argument-type]


def _infer_scalar_kinds(
    body: list[ast.stmt],
    *,
    float_scalar_names: Collection[str],
    tensor_dtypes: Mapping[str, str],
) -> _CanonicalKinds:
    """Forward dataflow over every assignment of the kernel body.  A name
    keeps a kind only if all of its assignments agree (loop-carried fp32
    accumulators do; anything assigned a non-fp32 value, destructured or
    augmented is unknown).  The names a masked select may reach are
    recorded separately (``_select_derived_names``)."""
    kinds = _CanonicalKinds()
    kinds.selected = _select_derived_names(body)
    for name in float_scalar_names:
        kinds.kinds[_canonical(name)] = "fp32"
    seeded = set(kinds.kinds)
    unknown: set[str] = set()
    assigns = [
        node for stmt in body for node in ast.walk(stmt) if isinstance(node, ast.stmt)
    ]
    # Two rounds: the first defines in program order, the second re-derives
    # every assignment with the complete map and records disagreements.
    for _round in range(2):
        for stmt in assigns:
            if isinstance(stmt, ast.AugAssign):
                unknown.update(_canonical(name) for name in _names_bound_by(stmt))
                continue
            if not isinstance(stmt, ast.Assign):
                continue
            target = stmt.targets[0] if len(stmt.targets) == 1 else None
            if not isinstance(target, ast.Name):
                unknown.update(_canonical(name) for name in _names_bound_by(stmt))
                continue
            name = _canonical(target.id)
            if name in seeded:
                # A reassigned scalar argument is no longer the argument.
                unknown.add(name)
                continue
            kind = _expression_kind(stmt.value, kinds, tensor_dtypes)
            if name not in kinds.kinds:
                kinds.kinds[name] = kind
            elif kinds.kinds[name] != kind:
                unknown.add(name)
    for name in unknown:
        kinds.kinds[name] = None
    return kinds


def _is_unit_constant(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float)) and node.value == 1
    # ``cutlass.Float32(1.0) / x``: the reciprocal the flash emitters spell.
    return (
        isinstance(node, ast.Call)
        and ast.unparse(node.func) == _FP32_DTYPE_STR
        and len(node.args) == 1
        and not node.keywords
        and _is_unit_constant(node.args[0])
    )


def _exact_division_candidates(
    loop_body: list[ast.stmt],
    loop_target: str | None,
    kinds: _CanonicalKinds,
) -> dict[str, list[ast.Assign]]:
    """``v_N = x / d`` statements anywhere in ``loop_body`` whose divisor is
    loop-invariant and resolves to a root name defined outside the loop,
    with both operands proved fp32 scalars and a dividend no per-thread
    select reaches (``_select_derived_names``).  Maps root -> statements."""
    assigns_by_target, all_written = _collect_assigns_in_body(loop_body)
    invariant_cache: dict[str, bool] = {}
    found: dict[str, list[ast.Assign]] = {}
    for stmt in loop_body:
        for node in ast.walk(stmt):
            if not (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id.startswith("v_")
            ):
                continue
            value = node.value
            if not (
                isinstance(value, ast.BinOp)
                and isinstance(value.op, ast.Div)
                and isinstance(value.right, ast.Name)
            ):
                continue
            if _is_unit_constant(value.left):
                continue
            ctx = _EXACT_CTX[0]
            if ctx is not None and _canonical(node.targets[0].id) not in ctx.reads:
                continue
            if not _is_loop_invariant(
                value.right.id,
                assigns_by_target,
                all_written,
                loop_target,
                invariant_cache,
                set(),
            ):
                continue
            root = _resolve_root_name(
                value.right.id, assigns_by_target, all_written, set()
            )
            if root is None or kinds.get(root) != "fp32":
                continue
            if _kind(value.left, kinds) != "fp32":  # pyrefly: ignore[bad-argument-type]
                continue
            if kinds.select_derived(value.left):
                continue
            found.setdefault(root, []).append(node)
    return found


EXACT_DIV_FAST_HELPER = "_cute_exact_div_fast_f32"
EXACT_DIV_REDO_HELPER = "_cute_exact_div_redo_f32"
# Marks the ``if`` holding a redo copy so the walks never descend into it.
EXACT_DIV_REDO_ATTR = "_helion_cute_exact_division_redo"
_REDO_COUNTER: list[int] = [0]

# Call forms a loop may contain and still be re-executed verbatim: pure
# arithmetic and conversions, plain loads, plain stores and the per-iteration
# Python lists the store packets are built from.  Anything else (barriers,
# shuffles, reductions, atomics, async copies, unknown helpers, any other
# ``cutlass.`` member such as ``cutlass.printf`` or ``cutlass.inttoptr``)
# makes the loop ineligible for the redo form; its sites keep the plain
# division unless a child loop can take them.
_REDO_PURE_PREFIXES = ("cute.math.",)
# ``cutlass.<Type>(...)`` conversions plus these lowercase members.
_REDO_CUTLASS_CALLS = frozenset(
    {"range", "range_constexpr", "const_expr", "select_", "min", "max"}
)
_REDO_IR_CALLS = frozenset({"ir.VectorType.get"})
_REDO_CUTE_ARCH_CALLS = frozenset(
    {
        "load",
        "fmax",
        "fmin",
        "thread_idx",
        "block_idx",
        "block_dim",
        "grid_dim",
        "rcp_approx",
        "fma_packed_f32x2",
        "mul_packed_f32x2",
        "add_packed_f32x2",
        "cvt_f32_tf32",
    }
)
_REDO_HELPER_PREFIXES = (
    "_cute_load_",
    "_cute_scalar_policy_load",
    "_cute_store_",
    "_cute_exact_div_",
    "_cute_join_cast",
    "_cute_copysign",
    "_cute_python_mod",
    "_cute_fp8",
    "_cute_float4_",
    "_cute_bfloat16_",
    "_cute_signed_bitfield",
    "_cute_inline_asm_elementwise",
)
_REDO_LOAD_HELPER_PREFIXES = ("_cute_load_", "_cute_scalar_policy_load")
_REDO_STORE_HELPER_PREFIX = "_cute_store_"
_REDO_PYTHON_BUILTINS = frozenset({"range", "int", "float", "len"})
_REDO_LIST_MUTATORS = frozenset(
    {"append", "extend", "insert", "pop", "remove", "clear"}
)
_REDO_PURE_METHODS = frozenset({"bitcast", "to"})
_PYTHON_OBJECT_RHS = (
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    ast.ListComp,
    ast.DictComp,
    ast.SetComp,
    ast.GeneratorExp,
    ast.Lambda,
)


class _ExactContext:
    """Kernel-level facts the redo eligibility test needs."""

    def __init__(
        self,
        body: list[ast.stmt],
        *,
        tensor_args: Collection[str],
        argument_names: Collection[str],
        disjoint_pairs: Collection[frozenset[str]],
    ) -> None:
        self.body = body
        # Names the kernel reads (canonical): a quotient nobody reads is a
        # dead pure assignment the DCE sub-pass removes; rewriting it would
        # keep it alive as a helper call.
        self.reads = {_canonical(name) for name in _collect_all_name_reads(body)}
        self.tensor_args = set(tensor_args)
        self.argument_names = set(argument_names)
        self.disjoint_pairs = set(disjoint_pairs)
        self.rmem: set[str] = set()
        self.python_objects: set[str] = set()
        for stmt in body:
            for node in ast.walk(stmt):
                if not (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                ):
                    continue
                name = _canonical(node.targets[0].id)
                value = node.value
                if (
                    isinstance(value, ast.Call)
                    and ast.unparse(value.func) == "cute.make_rmem_tensor"
                ):
                    self.rmem.add(name)
                elif isinstance(value, _PYTHON_OBJECT_RHS):
                    self.python_objects.add(name)


_EXACT_CTX: list[_ExactContext | None] = [None]


def _new_redo_names() -> tuple[str, str]:
    index = _REDO_COUNTER[0]
    _REDO_COUNTER[0] += 1
    return f"{EXACT_DIV_PREFIX}min_{index}", f"{EXACT_DIV_PREFIX}max_{index}"


# The DSL preprocessor outlines a ``for`` over ``range`` / ``cutlass.range`` /
# ``range_dynamic`` and an ``if`` / ``while`` on a dynamic test into a region
# function: the names such a region binds for the first time are not visible
# after it (only names that already existed are yielded back).  Everything
# else -- ``cutlass.range_constexpr`` and any non-``range`` iterable, an
# ``if`` / ``while`` on ``cutlass.const_expr(...)``, a ``with`` -- stays
# Python, runs while tracing, and keeps its bindings like any Python block.
_STAGED_RANGE_CALLS = frozenset({"range", "range_dynamic"})
_UNROLLED_TESTS = frozenset({"const_expr", "target_version"})


def _is_unrolled_region(stmt: ast.stmt) -> bool:
    """Whether Python itself runs ``stmt`` while tracing, so the names it
    binds stay visible afterwards (see the comment above)."""
    if isinstance(stmt, ast.With):
        return True
    if isinstance(stmt, ast.For):
        func = stmt.iter.func if isinstance(stmt.iter, ast.Call) else None
        if isinstance(func, ast.Name):
            return func.id not in _STAGED_RANGE_CALLS
        if isinstance(func, ast.Attribute):
            return func.attr not in _STAGED_RANGE_CALLS
        return True
    if isinstance(stmt, (ast.If, ast.While)):
        func = stmt.test.func if isinstance(stmt.test, ast.Call) else None
        if isinstance(func, ast.Name):
            return func.id in _UNROLLED_TESTS
        if isinstance(func, ast.Attribute):
            return func.attr in _UNROLLED_TESTS
    return False


def _bound_before(
    body: list[ast.stmt], target: ast.stmt, *, bound: Collection[str] = ()
) -> tuple[set[str], bool] | None:
    """Canonical names defined when ``target`` starts, under the DSL's
    staging rules: ``bound`` (the kernel's parameters, bound before any
    statement), the statements before it in program order, the headers of
    the loops enclosing it, the bindings of every unrolled region
    (``_is_unrolled_region``) that ends before it, and -- for a staged
    region that ends before it -- only the names that already existed
    before that region.  Also reports whether every statement enclosing
    ``target`` is a ``for``.  ``None`` when ``target`` is not in ``body``.

    Under-approximating this set is unsound (a loop-carried accumulator
    the redo form would not roll back; a parameter the loop rebinds is one
    too), over-approximating it fails closed (a save of an unbound name is
    a trace-time ``NameError``); where the DSL's rule is uncertain the walk
    keeps the name."""

    def walk(stmts: list[ast.stmt], defined: set[str]) -> tuple[bool, bool]:
        for stmt in stmts:
            if stmt is target:
                return True, True
            if isinstance(stmt, (ast.For, ast.While, ast.If, ast.With)):
                # Always a copy: when the target is found inside, this
                # level's ``defined`` is replaced wholesale by ``inner``
                # (clearing a shared set would empty both).
                inner = set(defined)
                if isinstance(stmt, ast.For):
                    inner.update(
                        _canonical(n.id)
                        for n in ast.walk(stmt.target)
                        if isinstance(n, ast.Name)
                    )
                for nested in (stmt.body, getattr(stmt, "orelse", [])):
                    found, only_for = walk(nested, inner)
                    if found:
                        defined.clear()
                        defined.update(inner)
                        return True, only_for and isinstance(stmt, ast.For)
                if _is_unrolled_region(stmt):
                    defined.update(inner)
            else:
                defined.update(_canonical(n) for n in _names_bound_by(stmt))
        return False, True

    defined: set[str] = {_canonical(name) for name in bound}
    found, only_for = walk(body, defined)
    if not found:
        return None
    return defined, only_for


def _call_path(node: ast.Call) -> str:
    return ast.unparse(node.func)


def _pointer_tensor(node: ast.expr) -> str | None:
    """The one tensor a pointer expression offsets (a masked tail spells
    ``t.iterator + i if ok else t.iterator + 0``), or None."""
    tensors = {
        sub.value.id if isinstance(sub.value, ast.Name) else None
        for sub in ast.walk(node)
        if isinstance(sub, ast.Attribute) and sub.attr == "iterator"
    }
    if len(tensors) != 1:
        return None
    return next(iter(tensors))


def _redo_memory(
    loop: ast.For, local_names: set[str], ctx: _ExactContext
) -> tuple[set[str], set[str], list[ast.expr]] | None:
    """(loaded tensors, stored tensors, address expressions) of ``loop``,
    or None when some memory access is not a recognized plain load/store."""
    loads: set[str] = set()
    stores: set[str] = set()
    addresses: list[ast.expr] = []
    recognized: list[ast.AST] = []
    for node in ast.walk(loop):
        if isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "load"
                and not node.args
                and not node.keywords
            ):
                tensor = _pointer_tensor(func.value)
                if tensor is None:
                    return None
                loads.add(tensor)
                addresses.append(func.value)
                recognized.append(func.value)
            elif (
                isinstance(func, ast.Attribute)
                and func.attr == "store"
                and len(node.args) == 1
            ):
                tensor = _pointer_tensor(func.value)
                if tensor is None:
                    return None
                stores.add(tensor)
                addresses.append(func.value)
                recognized.append(func.value)
            else:
                path = _call_path(node)
                is_load = path == "cute.arch.load" or path.startswith(
                    _REDO_LOAD_HELPER_PREFIXES
                )
                is_store = path.startswith(_REDO_STORE_HELPER_PREFIX)
                if is_load or is_store:
                    if not node.args:
                        return None
                    tensor = _pointer_tensor(node.args[0])
                    if tensor is None:
                        return None
                    (loads if is_load else stores).add(tensor)
                    addresses.append(node.args[0])
                    recognized.append(node.args[0])
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            base = _canonical(node.value.id)
            if isinstance(node.ctx, ast.Store):
                if base not in ctx.rmem:
                    return None
                stores.add(base)
                addresses.append(node.slice)
            elif base in ctx.rmem:
                loads.add(base)
                addresses.append(node.slice)
            elif base not in local_names:
                # Shared memory, a Python object from outside the loop, ...
                return None
    # Every ``.iterator`` must belong to a recognized access.
    all_iterators = sum(
        1
        for node in ast.walk(loop)
        if isinstance(node, ast.Attribute) and node.attr == "iterator"
    )
    seen_iterators = sum(
        1
        for root in recognized
        for node in ast.walk(root)
        if isinstance(node, ast.Attribute) and node.attr == "iterator"
    )
    if all_iterators != seen_iterators:
        return None
    return loads, stores, addresses


def _redo_calls_allowed(loop: ast.For, local_names: set[str]) -> bool:
    for node in ast.walk(loop):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            if func.attr in _REDO_LIST_MUTATORS:
                if not (
                    isinstance(func.value, ast.Name)
                    and _canonical(func.value.id) in local_names
                ):
                    return False
                continue
            if func.attr in _REDO_PURE_METHODS or func.attr in {"load", "store"}:
                continue
        path = _call_path(node)
        if path.startswith("cute.arch."):
            if path[len("cute.arch.") :] not in _REDO_CUTE_ARCH_CALLS:
                return False
            continue
        if path.startswith("cutlass."):
            member = path[len("cutlass.") :]
            if member in _REDO_CUTLASS_CALLS or (
                "." not in member and member[:1].isupper()
            ):
                continue
            return False
        if path.startswith(_REDO_PURE_PREFIXES) or path in _REDO_IR_CALLS:
            continue
        if path in _REDO_PYTHON_BUILTINS:
            continue
        if path.startswith(_REDO_HELPER_PREFIXES):
            if path == "_cute_inline_asm_elementwise" and any(
                keyword.arg == "is_pure"
                and not (
                    isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                )
                for keyword in node.keywords
            ):
                return False
            continue
        return False
    return True


def _redo_statements_allowed(loop: ast.For) -> bool:
    for node in ast.walk(loop):
        if isinstance(node, ast.stmt) and not isinstance(
            node, (ast.For, ast.Assign, ast.AugAssign, ast.Expr, ast.Pass)
        ):
            return False
        if isinstance(node, _PYTHON_OBJECT_RHS) and not isinstance(node, ast.List):
            return False
        if isinstance(node, (ast.NamedExpr, ast.Yield, ast.YieldFrom, ast.Await)):
            return False
        if isinstance(node, ast.For) and (
            node.orelse or not isinstance(node.target, ast.Name)
        ):
            return False
        if isinstance(node, ast.Expr) and not isinstance(node.value, ast.Call):
            return False
    return True


def _tainted_names(loop: ast.For, seeds: set[str]) -> set[str]:
    """Canonical names of ``loop`` whose value depends on a seed."""
    tainted = set(seeds)
    changed = True
    while changed:
        changed = False
        for node in ast.walk(loop):
            if isinstance(node, (ast.Assign, ast.AugAssign)):
                reads = {_canonical(n) for n in _names_read(node.value)}
                if isinstance(node, ast.AugAssign):
                    reads |= {_canonical(n) for n in _names_read(node.target)}
                if reads & tainted:
                    for name in _names_bound_by(node):
                        canon = _canonical(name)
                        if canon not in tainted:
                            tainted.add(canon)
                            changed = True
    return tainted


def _redo_eligible(
    loop: ast.For,
    sites: list[ast.Assign],
    ctx: _ExactContext,
) -> set[str] | None:
    """The loop-carried names to roll back when ``loop`` can be re-executed
    verbatim, or None when it cannot (see the module section comment)."""
    before = _bound_before(ctx.body, loop, bound=ctx.argument_names)
    if before is None:
        return None
    bound_before, only_for = before
    if not only_for:
        return None
    if not _redo_statements_allowed(loop):
        return None
    # An augmented assignment through a subscript (``cache[k] += v``) reads
    # and rewrites a register fragment: the redo would accumulate into it a
    # second time and no name identifies it for the rollback.
    for node in ast.walk(loop):
        if isinstance(node, ast.AugAssign) and not isinstance(node.target, ast.Name):
            return None
    local_names = {_canonical(n) for n in _names_bound_by(loop)}
    if not _redo_calls_allowed(loop, local_names):
        return None
    memory = _redo_memory(loop, local_names, ctx)
    if memory is None:
        return None
    loads, stores, addresses = memory
    for loaded in loads:
        for stored in stores:
            if loaded == stored:
                return None
            if loaded in ctx.tensor_args and stored in ctx.tensor_args:
                if frozenset({loaded, stored}) not in ctx.disjoint_pairs:
                    return None
    # A quotient must reach no address, trip count or array index: the
    # speculative pass may be wrong there and would read or write off the
    # tensor before the redo could correct it.
    for site in sites:
        if not isinstance(site.value, ast.BinOp) or not isinstance(
            site.value.left, ast.Name
        ):
            return None
    tainted = _tainted_names(
        loop,
        {_canonical(site.targets[0].id) for site in sites},  # pyrefly: ignore
    )
    guarded: list[ast.expr] = list(addresses)
    for node in ast.walk(loop):
        if isinstance(node, ast.For):
            guarded.append(node.iter)
    for expression in guarded:
        if {_canonical(n) for n in _names_read(expression)} & tainted:
            return None
    # Roll back what the loop reads and updates (accumulators); its own
    # loop variables and the temporaries it defines before reading are
    # re-derived by the redo.
    loop_targets = {
        _canonical(name)
        for node in ast.walk(loop)
        if isinstance(node, ast.For)
        for name in _names_bound_by(node.target)  # pyrefly: ignore[bad-argument-type]
    }
    read_in_loop = {_canonical(name) for name in _names_read(loop)}
    # An augmented assignment reads its target (a Store context, which
    # ``_names_read`` does not count).
    read_in_loop |= {
        _canonical(name)
        for node in ast.walk(loop)
        if isinstance(node, ast.AugAssign)
        for name in _names_bound_by(node)
    }
    carried = (local_names & bound_before & read_in_loop) - loop_targets
    if carried & ctx.python_objects:
        return None
    return carried


def _rewrite_fast_division(
    stmt: ast.Assign, root: str, y_name: str, tracker: tuple[str, str]
) -> list[ast.stmt]:
    """``v = x / d`` -> the unguarded fast quotient plus the two running
    extrema of ``|x|`` the redo test reads after the loop."""
    value = stmt.value
    assert isinstance(value, ast.BinOp) and isinstance(value.left, ast.Name)
    dividend = value.left.id
    replacement = expr_from_string(
        f"{_FP32_DTYPE_STR}({EXACT_DIV_FAST_HELPER}({{left}}, {root}, {y_name}))",
        left=value.left,
    )
    assert isinstance(replacement, ast.expr)
    stmt.value = ast.copy_location(replacement, value)
    ast.fix_missing_locations(stmt)
    min_name, max_name = tracker
    return [
        statement_from_string(
            f"{min_name} = cute.arch.fmin({min_name}, cute.math.abs({dividend}), nan=True)"
        ),
        statement_from_string(
            f"{max_name} = cute.arch.fmax({max_name}, cute.math.abs({dividend}), nan=True)"
        ),
    ]


def _insert_after(body: list[ast.stmt], anchor: ast.stmt, new: list[ast.stmt]) -> None:
    """Insert ``new`` right after ``anchor`` in the body list that holds it."""
    for stmts in _bodies_containing(body, anchor):
        index = next(i for i, stmt in enumerate(stmts) if stmt is anchor)
        stmts[index + 1 : index + 1] = new
        return
    raise AssertionError("anchor statement not found")


def _bodies_containing(
    body: list[ast.stmt], anchor: ast.stmt
) -> Iterator[list[ast.stmt]]:
    if any(stmt is anchor for stmt in body):
        yield body
        return
    for stmt in body:
        for attr in ("body", "orelse"):
            nested = getattr(stmt, attr, None)
            if isinstance(nested, list):
                yield from _bodies_containing(nested, anchor)


def _apply_redo_form(
    loop: ast.For,
    parent_body: list[ast.stmt],
    loop_idx: int,
    sites_by_root: dict[str, list[ast.Assign]],
    names_by_root: dict[str, tuple[str, str, str]],
    carried: set[str],
) -> int:
    """Rewrite ``loop`` into its speculative form and append the redo copy.
    Returns the number of statements inserted before ``loop_idx``."""
    original = _deep_clone(loop)
    before: list[ast.stmt] = []
    tests: list[str] = []
    for root, sites in sites_by_root.items():
        y_name, lo_name, hi_name = names_by_root[root]
        tracker = _new_redo_names()
        min_name, max_name = tracker
        before.extend(
            (
                statement_from_string(f"{min_name} = cutlass.Float32(float('inf'))"),
                statement_from_string(f"{max_name} = cutlass.Float32(0.0)"),
            )
        )
        tests.append(
            f"{EXACT_DIV_REDO_HELPER}({min_name}, {max_name}, {lo_name}, {hi_name})"
        )
        for site in sites:
            trackers = _rewrite_fast_division(site, root, y_name, tracker)
            _insert_after(loop.body, site, trackers)
    restores: list[ast.stmt] = []
    for name in sorted(carried):
        save = f"{EXACT_DIV_PREFIX}save_{_REDO_COUNTER[0]}_{name}"
        before.append(statement_from_string(f"{save} = {name}"))
        restores.append(statement_from_string(f"{name} = {save}"))
    redo = statement_from_string(f"if {' | '.join(tests)}:\n    pass")
    assert isinstance(redo, ast.If)
    redo.body = [*restores, original]
    setattr(redo, EXACT_DIV_REDO_ATTR, True)
    ast.fix_missing_locations(redo)
    parent_body[loop_idx + 1 : loop_idx + 1] = [redo]
    parent_body[loop_idx:loop_idx] = before
    return len(before)


def _direct_child_loops(loop: ast.For) -> list[ast.For]:
    return [stmt for stmt in loop.body if isinstance(stmt, ast.For)]


def _contains(node: ast.AST, target: ast.AST) -> bool:
    return any(sub is target for sub in ast.walk(node))


_RedoPlan = list[tuple[ast.For, dict[str, list[ast.Assign]], set[str]]]


def _plan_redo_loops(
    loop: ast.For,
    sites_by_root: dict[str, list[ast.Assign]],
    ctx: _ExactContext,
) -> _RedoPlan:
    """The loops of ``loop``'s nest that take the redo form, each with the
    sites it covers and the names its redo copy rolls back: ``loop`` itself
    when it can be re-executed verbatim, else (recursively) the child loops
    that can.  A site in no such loop keeps the plain division.  Decides
    only; nothing is rewritten."""
    all_sites = [site for sites in sites_by_root.values() for site in sites]
    carried = _redo_eligible(loop, all_sites, ctx)
    if carried is not None:
        return [(loop, sites_by_root, carried)]
    plan: _RedoPlan = []
    for child in _direct_child_loops(loop):
        child_sites = {
            root: [site for site in sites if _contains(child, site)]
            for root, sites in sites_by_root.items()
        }
        child_sites = {root: sites for root, sites in child_sites.items() if sites}
        if child_sites:
            plan.extend(_plan_redo_loops(child, child_sites, ctx))
    return plan


def _hoist_exact_divisions_in_for(
    for_node: ast.For,
    parent_body: list[ast.stmt],
    for_idx: int,
    live: dict[str, tuple[str, str, str]],
    kinds: _CanonicalKinds,
) -> int:
    """Hoist the divisor-only work of every candidate division of one
    ``for`` loop whose sites some loop of the nest takes in the redo form;
    ``live`` shares the hoisted names with later sibling loops dividing by
    the same, not yet rebound, root.  Returns the number of statements
    inserted before ``for_idx``.  A loop none of whose sites can be placed
    is left exactly as it is (no hoist, no helper: the plain render)."""
    loop_target = for_node.target.id if isinstance(for_node.target, ast.Name) else None
    candidates = _exact_division_candidates(for_node.body, loop_target, kinds)
    if not candidates:
        return 0
    ctx = _EXACT_CTX[0]
    assert ctx is not None
    plan = _plan_redo_loops(for_node, candidates, ctx)
    placed_roots = {
        root for _loop, sites_by_root, _carried in plan for root in sites_by_root
    }
    inserted = 0
    names_by_root: dict[str, tuple[str, str, str]] = {}
    for root in candidates:
        if root not in placed_roots:
            continue
        names = live.get(root)
        if names is None:
            names = _new_exact_names()
            y_name, lo_name, hi_name = names
            for decl in (
                statement_from_string(f"{y_name} = 1.0 / {root}"),
                statement_from_string(
                    f"{lo_name}, {hi_name} = {EXACT_DIV_BOUNDS_HELPER}({root})"
                ),
            ):
                parent_body.insert(for_idx + inserted, decl)
                inserted += 1
            live[root] = names
        names_by_root[root] = names
    for loop, sites_by_root, carried in plan:
        # Each rewrite shifts the statements after it in its own body list;
        # the loop's position is taken when its turn comes.
        body = next(_bodies_containing(parent_body, loop))
        loop_idx = next(i for i, stmt in enumerate(body) if stmt is loop)
        before = _apply_redo_form(
            loop, body, loop_idx, sites_by_root, names_by_root, carried
        )
        if body is parent_body:
            inserted += before
    return inserted


def _hoist_exact_divisions_in_body(
    body: list[ast.stmt], kinds: _CanonicalKinds
) -> list[ast.stmt]:
    """Outer-in walk (as ``_hoist_in_body``): the outermost ``for`` whose body
    keeps the divisor invariant receives the hoist, nested bodies are then
    visited for divisors that are invariant only deeper down."""
    if _is_placeholder_body(body):
        return body
    live: dict[str, tuple[str, str, str]] = {}
    i = 0
    while i < len(body):
        stmt = body[i]
        if isinstance(stmt, ast.For) and not stmt.orelse:
            inserted = _hoist_exact_divisions_in_for(stmt, body, i, live, kinds)
            i += inserted + 1
        else:
            i += 1
        for bound in _names_bound_by(stmt):
            live.pop(_canonical(bound), None)
            live.pop(bound, None)
    for stmt in body:
        if getattr(stmt, EXACT_DIV_REDO_ATTR, False):
            # A redo copy re-executes the original loop verbatim.
            continue
        if isinstance(stmt, (ast.For, ast.While, ast.If, ast.With, ast.FunctionDef)):
            stmt.body = _hoist_exact_divisions_in_body(stmt.body, kinds)
        if isinstance(stmt, (ast.For, ast.While, ast.If)):
            stmt.orelse = _hoist_exact_divisions_in_body(stmt.orelse, kinds)
    return body


def has_hoistable_fp32_divisions(graphs: Iterable[torch.fx.Graph]) -> bool:
    """Whether some traced fp32 ``aten.div`` divides by a scalar (a Python
    float argument, a symbolic float) or by a tensor of fewer elements than
    its result (a row value broadcast over the row): the divisions the
    exact hoist can take, so ``cute_exact_div_form`` has something to pick.
    The AST pass still proves loop invariance per site; this is only the
    gate that exposes the knob.  A kernel can pass it and still have no
    eligible site (the divisor is loop-carried at every loop, or the config
    creates no loop around the division): both forms then render
    identically and the knob is inert for that kernel -- one more
    autotuner dimension, never a different result."""
    for graph in graphs:
        for node in graph.nodes:
            if node.op != "call_function" or node.target not in (
                torch.ops.aten.div.Tensor,
                torch.ops.aten.div.Scalar,
            ):
                continue
            result = node.meta.get("val")
            if (
                not isinstance(result, torch.Tensor)
                or result.dtype is not torch.float32
            ):
                continue
            if len(node.args) < 2:
                continue
            divisor = node.args[1]
            if not isinstance(divisor, torch.fx.Node):
                return True  # a Python number
            value = divisor.meta.get("val")
            if not isinstance(value, torch.Tensor):
                return True  # a symbolic float
            if value.dtype is torch.float32 and (
                value.ndim < result.ndim
                or any(
                    d == 1 and r != 1
                    for d, r in zip(value.shape, result.shape, strict=False)
                )
            ):
                return True
    return False


def _has_division_statement(body: list[ast.stmt]) -> bool:
    """Whether some ``v_N = a / name`` statement exists at all: the only
    shape the exact hoist rewrites, so a body without one is left untouched
    (not even de-aliased)."""
    for stmt in body:
        for node in ast.walk(stmt):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id.startswith("v_")
                and isinstance(node.value, ast.BinOp)
                and isinstance(node.value.op, ast.Div)
                and isinstance(node.value.right, ast.Name)
            ):
                return True
    return False


def hoist_exact_invariant_divisions(
    body: list[ast.stmt],
    rename_groups: Mapping[str, str] | None = None,
    *,
    float_scalar_names: Collection[str] = (),
    tensor_dtypes: Mapping[str, str] | None = None,
    argument_names: Collection[str] = (),
    proven_disjoint_tensor_pairs: Collection[frozenset[str]] = (),
) -> list[ast.stmt]:
    """Sub-pass 2' (exact arithmetic): see the section comment above.

    Runs in canonical-aware invariance mode with ``rename_groups`` (the
    pre-rename -> canonical map ``hoist_loop_invariant_recips`` takes, so a
    loop-carried update spelled through a to-be-renamed alias counts as a
    write of the divisor).  ``argument_names`` (every kernel parameter:
    bound before any statement, so one a sweep rebinds is rolled back) and
    ``proven_disjoint_tensor_pairs`` feed the redo eligibility test; a
    division in no loop the redo form can take is left as written.  The
    ``plain`` form of ``cute_exact_div_form`` skips this pass altogether.

    The caller runs this LAST among the structural passes (right before the
    rename): the Markstein helper calls, the fmin/fmax trackers and the
    redo ``if`` are not statement shapes the sweep matchers
    (``split_rolled_vector_sums``, the pipelined and packed loads, the
    cluster pairs, the scalar epilogue) recognize, so running earlier would
    silently switch those passes off for the configs that enable them --
    and, for ``cute_independent_reduction``, change the summation order.
    Running last, the redo render keeps the structure the plain render has.
    ``HELION_DISABLE_HOIST_RECIP`` turns this pass off with the others.
    """
    import os

    if os.environ.get("HELION_DISABLE_HOIST_RECIP"):
        return body
    if not _has_division_statement(body):
        return body
    _EXACT_COUNTER[0] = 0
    _REDO_COUNTER[0] = 0
    tensor_dtypes = tensor_dtypes or {}
    global _RENAME_GROUPS
    prev_renames = _RENAME_GROUPS
    prev_canonical = _USE_CANONICAL_INVARIANCE[0]
    previous = _EXACT_CTX[0]
    _RENAME_GROUPS = dict(rename_groups or {})
    _USE_CANONICAL_INVARIANCE[0] = True
    try:
        # Codegen shares statement objects between a reduction sweep and
        # its rematerialized store sweep; each position gets its own copy
        # before any in-place rewrite.
        _dealias_shared_nodes(body)
        kinds = _infer_scalar_kinds(
            body, float_scalar_names=float_scalar_names, tensor_dtypes=tensor_dtypes
        )
        _EXACT_CTX[0] = _ExactContext(
            body,
            tensor_args=tensor_dtypes,
            argument_names=argument_names,
            disjoint_pairs=proven_disjoint_tensor_pairs,
        )
        return _hoist_exact_divisions_in_body(body, kinds)
    finally:
        _EXACT_CTX[0] = previous
        _USE_CANONICAL_INVARIANCE[0] = prev_canonical
        _RENAME_GROUPS = prev_renames


def hoist_loop_invariant_recips(
    body: list[ast.stmt],
    rename_groups: dict[str, str] | None = None,
    *,
    fast_math: bool,
) -> list[ast.stmt]:
    """Apply the hoist passes to a list of kernel-body statements.

    Runs four sub-passes (2, 3 and 4 only when ``fast_math``, the setting,
    allows rounding to depend on the config; the bit-exact fp32 hoist 2',
    ``hoist_exact_invariant_divisions``, is a separate entry point the
    caller runs after the structural passes):

      1. Inline pure SSA alias chains (``mi_copy_1 = mi`` →
         direct use of ``mi``) so subsequent passes see clean root names.
      2. Hoist ``x / scalar`` divisions where the divisor is
         loop-invariant (transitively) into ``inv = 1.0 / scalar +
         x * inv``.
      3. Share one ``inv = 1.0 / d`` between ``n >= 2`` divisions by the
         same divisor value inside one straight-line block (the residual
         divisions whose divisor is recomputed every iteration, e.g.
         ``swiglu_bwd``'s three quotients by ``exp(x) + 1``).  Same
         rounding as sub-pass 2 -- see the module docstring for the
         <= 1 ulp argument.
      4. Hoist ``(A - INV) * CONST`` patterns where INV is
         loop-invariant — emits ``INV_scaled = INV * CONST`` outside
         and rewrites the inner expression to ``A * CONST - INV_scaled``
         (an FMA-friendly form).  Distributing the product rounds
         differently, and loses precision when ``A`` is close to a large
         ``INV`` (softmax's ``x - max``); like the reciprocals it fires
         only in the loops a config creates.

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
        if fast_math:
            body = _hoist_in_body(body)
            # 3. Share reciprocals between the divisions sub-pass 2 could
            #    not hoist (divisor recomputed per iteration) — same
            #    canonical-aware mode so a rebind through a to-be-renamed
            #    alias ends the sharing group.
            body = _share_repeated_recips_in_body(body)
            # 4. Hoist scaled-sub patterns — same canonical-aware mode.
            body = _hoist_scaled_subs_in_body(body)
        # 5. DCE dead pure assigns left behind by the rewrites
        #    (e.g. ``v_10 = v_9 - mi`` now unused because the Mult was
        #    rewritten to read ``v_9`` directly).
        body = _dce_pure_assigns(body)
    finally:
        _USE_CANONICAL_INVARIANCE[0] = False
        _RENAME_GROUPS = prev_renames
    return body
