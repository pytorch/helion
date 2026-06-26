"""AST peephole that eliminates redundant bitcast/cast work across two
adjacent constexpr V-loops in CuTe two-pass reductions.

Online softmax / two-pass reductions emit a SINGLE vec load
(``_tile_unroll_vec_*``) at the top of the per-tile body, then run TWO
constexpr V-loops:

  V-loop 1 (max pass)
      for vec_lane in cutlass.range_constexpr(V):
          ...
          values = bitcast(_tile_unroll_vec_*[vec_lane])
          ...
          _helion_vfold_acc_0 = max(..., Float32(values))
  scalar reduce + mi/di update (warp_reduce_max → mi_next → ...)
  V-loop 2 (sum pass)
      for vec_lane in cutlass.range_constexpr(V):
          ...
          values = bitcast(_tile_unroll_vec_*[vec_lane])     # REDUNDANT
          ...
          _helion_vfold_acc_1 += exp2((Float32(values) - mi_next) * 1.44)

The two V-loops always share the SAME prefix of statements (same induction
variable, same `vec_lane` reads of the same hoist var, same masks).  The
SASS scheduler treats them as independent dependency chains but still
re-issues the bitcast/cast/IfExp guards twice — and the second V-loop's
register pressure waits on the first V-loop's results to commit.

This pass caches the LAST common per-V-lane scalar (always ``values`` in
practice) into a small ``cute.make_fragment(V, dtype)``.  V-loop 2's
identical prefix is then stripped — its `values` becomes a cache read.
The bitcast and IfExp guard run ONCE per outer iter instead of twice,
and the second V-loop's exp2 chain can issue immediately because its
input is already register-resident.

Conservatively applied: only triggers when

  1. Two adjacent constexpr V-loops with same induction variable and
     same constexpr trip count.
  2. The two loop bodies have an identical statement prefix (matched by
     textual ``ast.unparse``) of length >= 1.
  3. The LAST statement of the identical prefix assigns to a single Name.

If any condition fails the pass leaves the body untouched.
"""

from __future__ import annotations

import ast
import os
import re

from ..ast_extension import statement_from_string
from ._ast_pass_utils import _assignment_lhs_name
from ._ast_pass_utils import _names_read


def _is_constexpr_v_loop(node: ast.stmt) -> tuple[ast.For, str, int] | None:
    """If ``node`` is ``for X in cutlass.range_constexpr(V): ...`` with a
    constant V and a plain Name target, return ``(loop, X, V)``.
    """
    if not isinstance(node, ast.For):
        return None
    if not isinstance(node.target, ast.Name):
        return None
    iter_node = node.iter
    if not (
        isinstance(iter_node, ast.Call)
        and isinstance(iter_node.func, ast.Attribute)
        and iter_node.func.attr == "range_constexpr"
        and isinstance(iter_node.func.value, ast.Name)
        and iter_node.func.value.id == "cutlass"
    ):
        return None
    if not iter_node.args:
        return None
    arg = iter_node.args[0]
    if not (isinstance(arg, ast.Constant) and isinstance(arg.value, int)):
        return None
    return node, node.target.id, arg.value


def _names_written(stmt: ast.stmt) -> set[str]:
    written: set[str] = set()
    if isinstance(stmt, ast.Assign):
        for target in stmt.targets:
            if isinstance(target, ast.Name):
                written.add(target.id)
    return written


def _infer_cache_dtype(rhs: ast.expr) -> str | None:
    """Detect the dtype wrapper for the RHS we want to cache.

    For ``cutlass.Uint16(...).bitcast(cutlass.Float16) if mask else cutlass.Float16(0)``
    we want the bitcast target — Float16.  For ``cutlass.Float32(x)`` we
    want Float32.  Returns None when no recognised dtype wrapper is found.

    Promote fp16/bf16 to fp32: register pressure is the same (cute fp16
    is stored 32b in a register anyway), and the downstream uses cast
    to fp32 immediately — so caching as fp32 elides the cast in V-loop 2.
    """
    if isinstance(rhs, ast.IfExp):
        # body / orelse share the same outer wrapper.
        return _infer_cache_dtype(rhs.body)
    if isinstance(rhs, ast.Call):
        func = rhs.func
        # bitcast(cutlass.Float16) chain.
        if isinstance(func, ast.Attribute) and func.attr == "bitcast":
            if rhs.args and isinstance(rhs.args[0], ast.Attribute):
                arg = rhs.args[0]
                text = ast.unparse(arg)
                # Promote fp16/bf16 to fp32 for the cache.
                if text in ("cutlass.Float16", "cutlass.BFloat16"):
                    return "cutlass.Float32"
                if text.startswith("cutlass."):
                    return text
            if rhs.args and isinstance(rhs.args[0], ast.Name):
                return rhs.args[0].id
        # cutlass.Float32(x) / cutlass.Float16(x) / cutlass.BFloat16(x)
        if isinstance(func, ast.Attribute):
            text = ast.unparse(func)
            if text in ("cutlass.Float16", "cutlass.BFloat16"):
                return "cutlass.Float32"
            if text.startswith(("cutlass.Float", "cutlass.BFloat")):
                return text
    return None


_CACHE_COUNTER: list[int] = [0]


def _new_cache_name() -> str:
    name = f"_helion_vmerge_cache_{_CACHE_COUNTER[0]}"
    _CACHE_COUNTER[0] += 1
    return name


def _try_merge_pair(
    new_body: list[ast.stmt],
    first_idx: int,
    second_idx: int,
) -> bool:
    """Try to merge two constexpr V-loops at positions ``first_idx`` /
    ``second_idx`` in ``new_body`` (with optional scalar code between).

    Returns True if the merge fired (and ``new_body`` was mutated), False
    otherwise.  On failure the body is left untouched.
    """
    first = new_body[first_idx]
    second = new_body[second_idx]
    first_match = _is_constexpr_v_loop(first)
    second_match = _is_constexpr_v_loop(second)
    if first_match is None or second_match is None:
        return False
    first_loop, first_var, first_trip = first_match
    second_loop, second_var, second_trip = second_match
    if first_trip != second_trip:
        return False
    if first_trip <= 1:
        return False

    body1 = first_loop.body
    body2 = second_loop.body
    if not body1 or not body2:
        return False

    # Collect names written by the inter-loop scalar code: any of these
    # being read by V-loop 1's stripped prefix would mean V-loop 2's
    # prefix sees DIFFERENT values, so the strip is unsafe.  In practice
    # the inter-loop code writes ``local_amax``, ``v_0``, ``v_1`` etc.,
    # none of which the V-loop prefix reads — but be defensive.
    inter_loop_writes: set[str] = set()
    for k in range(first_idx + 1, second_idx):
        for sub in ast.walk(new_body[k]):
            if isinstance(sub, ast.Assign):
                for target in sub.targets:
                    if isinstance(target, ast.Name):
                        inter_loop_writes.add(target.id)

    # Canonicalise V-loop 2's induction var name to match V-loop 1's so that
    # statement-level text matching works when the autotuner picks
    # different per-sweep counters (e.g. vec_lane_1 vs vec_lane_2).
    text2_canonical: list[str] = []
    for s in body2:
        text = ast.unparse(s)
        if second_var != first_var:
            text = re.sub(rf"\b{re.escape(second_var)}\b", first_var, text)
        text2_canonical.append(text)

    # Find longest identical prefix between body1 and body2 (after canonical
    # rename of V-loop 2's induction var).  An identical statement has the
    # same text and is an Assign producing a Name target (so we can take its
    # LHS).
    text1 = [ast.unparse(s) for s in body1]
    common_prefix_len = 0
    for i in range(min(len(text1), len(text2_canonical))):
        if text1[i] != text2_canonical[i]:
            break
        common_prefix_len = i + 1

    if common_prefix_len == 0:
        return False

    # Determine which names V-loop 2's suffix (after the strip) actually
    # READS from the stripped prefix.  Those are the candidate cache
    # variables — caching anything else gains nothing (it'd be dead).
    suffix_reads_v2: set[str] = set()
    for i in range(common_prefix_len, len(body2)):
        suffix_reads_v2 |= _names_read(body2[i])
    prefix_writes_v2: set[str] = set()
    for i in range(common_prefix_len):
        prefix_writes_v2 |= _names_written(body2[i])
    candidate_names = suffix_reads_v2 & prefix_writes_v2

    # Walk backward from the prefix end to find the last identical Assign
    # statement that produces a NEEDED variable (one the suffix actually
    # consumes) AND has an inferable cache dtype.
    cache_stmt_idx: int | None = None
    cache_var: str | None = None
    cache_dtype: str | None = None
    for i in range(common_prefix_len - 1, -1, -1):
        stmt = body1[i]
        lhs = _assignment_lhs_name(stmt)
        if lhs is None:
            continue
        if lhs not in candidate_names:
            continue
        rhs = stmt.value if isinstance(stmt, ast.Assign) else None
        if rhs is None:
            continue
        dtype = _infer_cache_dtype(rhs)
        if dtype is None:
            continue
        cache_stmt_idx = i
        cache_var = lhs
        cache_dtype = dtype
        break

    if cache_stmt_idx is None or cache_var is None or cache_dtype is None:
        return False

    # We will REPLACE body2[cache_stmt_idx] with a cache read; the rest of
    # body2's prefix (mask/indices computation) we leave untouched so the
    # suffix's reads of those names still resolve.  Validate that body2's
    # cache_stmt_idx is itself an Assign producing the same LHS name as
    # body1's cache_stmt_idx (the common-prefix-text check already
    # implies this, but be explicit).
    body2_cache_stmt = body2[cache_stmt_idx]
    if _assignment_lhs_name(body2_cache_stmt) != cache_var:
        return False

    # Don't try to cache a constexpr-only expression (no V-dep): no perf
    # gain and could regress.  Heuristic: the cached stmt must read the
    # V-loop induction var (transitively through earlier stmts).  Check
    # whether any stmt in body1[0..cache_stmt_idx] either is the loop var
    # itself or transitively reads it.
    v_defined_names: set[str] = {first_var}
    for i in range(cache_stmt_idx + 1):
        s = body1[i]
        reads = _names_read(s)
        if reads & v_defined_names:
            v_defined_names.update(_names_written(s))
    if cache_var not in v_defined_names:
        return False

    # Verify: the cached stmt's RHS (and any earlier prefix stmt it
    # depends on transitively) must NOT read any name written by the
    # inter-loop scalar code.  Otherwise V-loop 1 would cache a stale
    # value before the inter-loop update.
    prefix_reads: set[str] = set()
    for i in range(common_prefix_len):
        prefix_reads |= _names_read(body1[i])
    if prefix_reads & inter_loop_writes:
        return False

    # ---- COMMIT PHASE ----
    cache_name = _new_cache_name()

    # 1. Build the cache fragment decl above V-loop 1.
    decl = statement_from_string(
        f"{cache_name} = cute.make_fragment({first_trip}, {cache_dtype})"
    )

    # 2. Rewrite V-loop 1: after the cache_stmt_idx assignment, write
    # ``cache_name[<first_var>] = <cache_dtype>(<cache_var>)``.  The
    # dtype wrap handles the fp16/bf16 → fp32 promotion (so V-loop 2 can
    # read fp32 directly without an extra cast).
    new_body1: list[ast.stmt] = []
    for i, s in enumerate(body1):
        new_body1.append(s)
        if i == cache_stmt_idx:
            new_body1.append(
                statement_from_string(
                    f"{cache_name}[{first_var}] = {cache_dtype}({cache_var})"
                )
            )
    first_loop.body = new_body1

    # 3. Rewrite V-loop 2: replace ONLY body2[cache_stmt_idx] (which
    # produces ``cache_var``) with a read from the cache.  The rest of
    # the prefix (e.g. mask_1 = indices_2 < N) is left untouched so the
    # suffix's reads still resolve.  Re-parse all statements using the
    # canonicalised text so induction-var renames are baked in.
    cache_read = statement_from_string(f"{cache_var} = {cache_name}[{first_var}]")
    new_body2: list[ast.stmt] = []
    for i in range(len(body2)):
        if i == cache_stmt_idx:
            new_body2.append(cache_read)
        else:
            text = text2_canonical[i]
            parsed = ast.parse(text).body[0]
            assert isinstance(parsed, ast.stmt)
            new_body2.append(parsed)
    second_loop.body = new_body2
    # Rebuild the V-loop 2's target to use first_var for consistency.
    if second_var != first_var:
        second_loop.target = ast.Name(id=first_var, ctx=ast.Store())
        ast.fix_missing_locations(second_loop)

    # 4. Insert the cache decl above V-loop 1.  (Must update second_idx
    # awareness for caller — but we mutate new_body in-place and the caller
    # re-scans, so just insert at first_idx.)
    new_body.insert(first_idx, decl)

    return True


def _merge_in_body(body: list[ast.stmt]) -> list[ast.stmt]:
    """Walk ``body`` recursively; for each pair of adjacent constexpr
    V-loops, try the merge.  Returns the (possibly modified) body.
    """
    # Recurse into nested bodies first.
    for stmt in body:
        if isinstance(stmt, (ast.For, ast.If, ast.With, ast.FunctionDef)):
            stmt.body = _merge_in_body(stmt.body)
        if isinstance(stmt, (ast.For, ast.If)):
            stmt.orelse = _merge_in_body(stmt.orelse)

    # Now look for V-loop pairs at THIS level.  The two V-loops do NOT
    # need to be adjacent — in online softmax the inter-loop region
    # contains the warp_reduce + mi/di scalar update.  We require the
    # inter-loop statements to not write to any name read by the cached
    # variable's RHS or by V-loop 1's stripped prefix.  This is checked
    # inside ``_try_merge_pair``.
    #
    # Repeat until no more merges fire (a single body could contain
    # multiple consecutive pairs across many tiles, but in practice the
    # strategy emits exactly one pair per outer iter so one pass suffices).
    new_body = list(body)
    changed = True
    while changed:
        changed = False
        # Find each pair of consecutive V-loops (with arbitrary scalar
        # code between them).  Walk forward and pair index k with the
        # next V-loop index after k.
        v_loop_indices: list[int] = []
        for k, s in enumerate(new_body):
            if _is_constexpr_v_loop(s) is not None:
                v_loop_indices.append(k)
        for pair_i in range(len(v_loop_indices) - 1):
            first_idx = v_loop_indices[pair_i]
            second_idx = v_loop_indices[pair_i + 1]
            # Skip if there's another V-loop between them (shouldn't happen
            # given how we built v_loop_indices, but be defensive).
            if any(
                _is_constexpr_v_loop(new_body[m]) is not None
                for m in range(first_idx + 1, second_idx)
            ):
                continue
            if _try_merge_pair(new_body, first_idx, second_idx):
                changed = True
                # Re-scan since indices have shifted (we inserted a decl
                # at first_idx, so v_loop indices have shifted by 1).
                break
    return new_body


_WARP_REDUCE_ATTRS = {
    "warp_reduction_sum",
    "warp_reduction_max",
    "warp_reduction_min",
    "warp_reduction",
}
_CUTLASS_NUMERIC_CASTS = {
    "cutlass.Float16",
    "cutlass.Float32",
    "cutlass.Float64",
    "cutlass.BFloat16",
}


def _looks_like_warp_reduce_call(node: ast.AST) -> bool:
    """True if ``node`` is a ``cute.arch.warp_reduction_*(...)`` call."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr not in _WARP_REDUCE_ATTRS:
        return False
    val = func.value
    return isinstance(val, ast.Attribute) and val.attr == "arch"


def _cast_dtype_name(call: ast.Call) -> str | None:
    """If ``call`` is ``cutlass.Float<N>(arg)`` return the cast's dtype text."""
    if not isinstance(call.func, ast.Attribute):
        return None
    text = ast.unparse(call.func)
    if text in _CUTLASS_NUMERIC_CASTS:
        return text
    return None


class _NameUseCountVisitor(ast.NodeVisitor):
    """Count uses of each Name (Load context) within a list of stmts."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.counts[node.id] = self.counts.get(node.id, 0) + 1


def _elide_double_casts_in_stmts(stmts: list[ast.stmt]) -> list[ast.stmt]:
    """Walk ``stmts`` and merge a redundant cast round-trip on
    warp_reduction results.

    Pattern (over CONSECUTIVE stmts):

        A = cutlass.Float<N>(cute.arch.warp_reduction_*(...))   # outer cast 1
        B = cutlass.Float<M>(A)                                   # outer cast 2

    When N != M and ``A`` has exactly ONE total use in the body (the
    ``Float<M>(A)`` line), the pair is replaced with:

        A = cutlass.Float<N>(cute.arch.warp_reduction_*(...))   # untouched
        B = cutlass.Float<M>(cute.arch.warp_reduction_*(...))   # skip A

    Strictly speaking, we want to ELIDE the Float<N> entirely when M
    matches the underlying type — but the underlying warp_reduction
    return dtype is determined by the hoist pass's acc dtype, which is
    ALWAYS cutlass.Float32 for fp16/bf16 inputs.  So when M == Float32
    we drop the Float<N> wrapper, yielding:

        A = cute.arch.warp_reduction_*(...)
        B = A
    """
    if not stmts:
        return stmts

    # Count uses across all stmts at this level (we only inline when A is
    # used exactly once — by the next stmt).
    counter = _NameUseCountVisitor()
    for s in stmts:
        counter.visit(s)
    total_use_counts = counter.counts

    new_stmts: list[ast.stmt] = []
    i = 0
    while i < len(stmts):
        stmt = stmts[i]
        consumed = False
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Call)
            and i + 1 < len(stmts)
        ):
            a_name = stmt.targets[0].id
            inner_cast = stmt.value
            inner_cast_dtype = _cast_dtype_name(inner_cast)
            # Inner expression must be a single-arg call whose argument
            # is a warp_reduction.
            if (
                inner_cast_dtype is not None
                and len(inner_cast.args) == 1
                and _looks_like_warp_reduce_call(inner_cast.args[0])
            ):
                next_stmt = stmts[i + 1]
                if (
                    isinstance(next_stmt, ast.Assign)
                    and len(next_stmt.targets) == 1
                    and isinstance(next_stmt.targets[0], ast.Name)
                    and isinstance(next_stmt.value, ast.Call)
                    and len(next_stmt.value.args) == 1
                    and isinstance(next_stmt.value.args[0], ast.Name)
                    and next_stmt.value.args[0].id == a_name
                ):
                    outer_cast_dtype = _cast_dtype_name(next_stmt.value)
                    if (
                        outer_cast_dtype is not None
                        and outer_cast_dtype == "cutlass.Float32"
                    ):
                        # ``A`` must be used exactly ONCE (the Float<M>(A))
                        # so dropping the inner cast doesn't break anything.
                        if total_use_counts.get(a_name, 0) == 1:
                            # Replace stmt: ``A = warp_reduction(...)``
                            new_a = ast.Assign(
                                targets=[ast.Name(id=a_name, ctx=ast.Store())],
                                value=inner_cast.args[0],
                            )
                            ast.copy_location(new_a, stmt)
                            new_stmts.append(new_a)
                            # Replace next_stmt: ``B = A`` (i.e. drop
                            # the outer cast since A is now fp32)
                            new_b = ast.Assign(
                                targets=next_stmt.targets,
                                value=ast.Name(id=a_name, ctx=ast.Load()),
                            )
                            ast.copy_location(new_b, next_stmt)
                            new_stmts.append(new_b)
                            i += 2
                            consumed = True
        if not consumed:
            new_stmts.append(stmt)
            i += 1
    return new_stmts


def _elide_double_casts_in_body(body: list[ast.stmt]) -> list[ast.stmt]:
    """Walk ``body`` recursively; apply ``_elide_double_casts_in_stmts``
    to every list of sibling statements.
    """
    for stmt in body:
        if isinstance(stmt, (ast.For, ast.If, ast.With, ast.FunctionDef)):
            stmt.body = _elide_double_casts_in_body(stmt.body)
        if isinstance(stmt, (ast.For, ast.If)):
            stmt.orelse = _elide_double_casts_in_body(stmt.orelse)
    return _elide_double_casts_in_stmts(body)


def merge_sibling_v_loops(body: list[ast.stmt]) -> list[ast.stmt]:
    """Apply the merge pass to a list of kernel-body statements.

    Returns the (possibly modified) body.  Safe to call on any kernel body
    — only rewrites when adjacent constexpr V-loops share an identical
    statement prefix.

    Set ``HELION_DISABLE_MERGE_VLOOPS=1`` to skip the pass (escape hatch
    for cases where the cache fragment causes a register-pressure
    regression).
    """
    if os.environ.get("HELION_DISABLE_MERGE_VLOOPS"):
        return body
    _CACHE_COUNTER[0] = 0
    result = _merge_in_body(body)
    # Also elide the double-cast on warp_reduction results.  See
    # ``_elide_double_casts_in_stmts`` docstring for the pattern matched.
    return _elide_double_casts_in_body(result)
