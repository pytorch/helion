"""AST peephole that software-pipelines per-iteration vec loads in CuTe
two-pass / single-pass reduction kernels.

The CuTe softmax / online-reduction inner loop currently emits::

    for tile_offset_X in range(START, END, STEP):
        for lane_X in range(LANE_REPS):
            LANE_BASE = tile_offset_X + thread_idx*V + lane_X*V
            LOAD_VAR = cute.arch.load(<addr depending on LANE_BASE>, VEC_TY)
            # ... compute body using LOAD_VAR and LANE_BASE ...

The whole iteration is data-dependent on the load result, so each warp
sees its iteration sequentially: issue load, wait for HBM to return,
compute, repeat.  At ~150-300 cycles per HBM round-trip, this is the
single largest scheduling stall on the inner loop after vec-hoist /
LICM / warp-reduce hoisting have squeezed the per-iter compute.

This pass software-pipelines the load by one stage: it pre-issues the
first iteration's load above the loop and, inside the loop body, issues
the NEXT iteration's load BEFORE the current iteration's compute body
runs::

    # Prologue: pre-issue iter 0 load
    _pipe_lane_base = START + thread_idx*V + 0*V
    _pipe_load = cute.arch.load(<addr using _pipe_lane_base>, VEC_TY)
    for tile_offset_X in range(START, END, STEP):
        for lane_X in range(LANE_REPS):
            # Snapshot current iter's prefetched values
            LANE_BASE = _pipe_lane_base
            LOAD_VAR = _pipe_load
            # Issue NEXT iter's load
            _pipe_lane_base = (tile_offset_X + STEP) + thread_idx*V + lane_X*V
            _pipe_load = cute.arch.load(<addr using _pipe_lane_base>, VEC_TY)
            # ... compute body uses LOAD_VAR + LANE_BASE (unchanged) ...

The masked-load form (``... if lane_base < N else ... idx 0``) already
guards against the speculative next-iter load reading out of bounds on
the final iteration: the mask falls through to a safe in-bounds index
0 read.

Correctness:
  * Prologue load is identical to the iter-0 load that the original
    code would have issued first thing inside the loop body.
  * Per-iter snapshot ``LOAD_VAR = _pipe_load`` is exactly the value
    issued in the prior iter's prefetch (or the prologue for iter 0).
  * The next-iter prefetch updates ``_pipe_*`` AFTER the snapshot, so
    the body reads the CORRECT (current iter's) values.
  * For the LAST iteration, the speculative prefetch reads beyond the
    end of the bounds-checked region — the masked-load form returns a
    safe in-bounds value which we never use (the loop ends after this
    iter).

The pass is conservative.  It only triggers when:

  1. A top-level ``for tile_offset in range(<int>, <int>, <int>)`` loop
     containing a single inner ``for lane in range(<int>)`` loop with
     LANE_REPS == 1.  (When LANE_REPS > 1 the prefetch update would
     corrupt the snapshot read by the next lane index in the same outer
     iter — see the in-source comment in ``_try_pipeline_one_outer_loop``.)
  2. The first two statements of the inner-lane body are a `lane_base`
     assignment (``LANE_BASE = tile_offset + ...``) followed by a
     vector load assignment (``LOAD_VAR = cute.arch.load(..., VecType)``).
  3. The load's address expression mentions LANE_BASE.
  4. The outer-loop trip count must be > 1 (otherwise there's no second
     iter to pipeline) and the loop must not be empty.

P19 (optional gate): an opt-in heuristic restricts pipelining to
loops whose body contains a **loop-carried scalar write** — an
assignment to a name that is defined in the OUTER scope before the
loop (a reduction accumulator like ``mi``/``di``).  This is the
structural marker that the load feeds a per-iter dependency chain
the SASS scheduler can't overlap on its own.

For pure stream-store sweeps (e.g. the consume sweep of two-pass
softmax), the SASS scheduler can already overlap consecutive iter
loads since slow ops like ``exp``/``div`` provide latency room.
In the standard microbench (5248/10240/12672/16384) pipelining
the consume sweep still measures slightly faster at the GPU level
than skipping it, so the gate is OFF by default.

The gate is provided as an opt-in for autotuner sweeps that want
to explore the "carried-only" alternative when a specific shape
shows a wall-clock regression that the default pipeline misses.

Set ``HELION_DISABLE_LOAD_PIPELINE=1`` to skip the pass entirely.
Set ``HELION_LOAD_PIPELINE_CARRIED_ONLY=1`` to enable the P19 gate
(only pipeline loops with a loop-carried scalar write).
Set ``HELION_LOAD_PIPELINE_ALL=1`` to force-disable the gate even
when the carried-only env var is set (used in test/debug scripts).
"""

from __future__ import annotations

import ast
import os

from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ._ast_pass_utils import _assignment_lhs_name
from ._ast_pass_utils import _names_read

_PIPE_COUNTER: list[int] = [0]


def _new_pipe_names() -> tuple[str, str]:
    """Allocate a fresh pair of pipeline shadow names."""
    idx = _PIPE_COUNTER[0]
    _PIPE_COUNTER[0] += 1
    return f"_pipe_lane_base_{idx}", f"_pipe_load_{idx}"


def _looks_like_outer_range_loop(
    stmt: ast.stmt,
) -> tuple[ast.For, str, ast.expr, ast.expr, ast.expr] | None:
    """If ``stmt`` is ``for X in range(START, END, STEP): ...`` with the
    target a plain Name, return ``(loop, X, START, END, STEP)``.

    Both 2-arg and 3-arg range forms are accepted.  The 2-arg form is
    treated as step=1 (a Constant(1) is synthesized).
    """
    if not isinstance(stmt, ast.For):
        return None
    if not isinstance(stmt.target, ast.Name):
        return None
    it = stmt.iter
    if not (
        isinstance(it, ast.Call)
        and isinstance(it.func, ast.Name)
        and it.func.id == "range"
    ):
        return None
    args = it.args
    if len(args) == 2:
        return stmt, stmt.target.id, args[0], args[1], ast.Constant(value=1)
    if len(args) == 3:
        return stmt, stmt.target.id, args[0], args[1], args[2]
    return None


def _looks_like_inner_lane_loop(
    stmt: ast.stmt,
) -> tuple[ast.For, str, int] | None:
    """If ``stmt`` is ``for LANE in range(<int>): ...`` with a single
    integer arg, return ``(loop, LANE, REPS)``.

    The integer arg may be a Python int Constant or a ``cutlass.Int32(N)``
    call wrapping a Constant.
    """
    if not isinstance(stmt, ast.For):
        return None
    if not isinstance(stmt.target, ast.Name):
        return None
    it = stmt.iter
    if not (
        isinstance(it, ast.Call)
        and isinstance(it.func, ast.Name)
        and it.func.id == "range"
    ):
        return None
    if len(it.args) != 1:
        return None
    reps = _const_int_value(it.args[0])
    if reps is None or reps <= 0:
        return None
    return stmt, stmt.target.id, reps


def _const_int_value(
    node: ast.expr,
    constexpr_values: dict[str, int] | None = None,
) -> int | None:
    """Extract a static integer from a Constant, ``cutlass.Int32(N)`` call,
    or a Name that's been declared as a module-level constexpr int.

    ``constexpr_values`` maps constexpr name -> static int (e.g.
    ``{"_BLOCK_SIZE_1": 128}``).
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"Int32", "Int64", "Uint32", "Uint64"}
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "cutlass"
        and len(node.args) == 1
    ):
        return _const_int_value(node.args[0], constexpr_values)
    if (
        isinstance(node, ast.Name)
        and constexpr_values is not None
        and node.id in constexpr_values
    ):
        return constexpr_values[node.id]
    return None


def _is_vec_load_call(node: ast.expr) -> bool:
    """True if ``node`` is ``cute.arch.load(<ptr>, ir.VectorType.get([V], ...))``.

    Matches both the raw call and the wrapped masked form used in some
    backends.  We also accept any 2-arg ``cute.arch.load(...)`` to keep
    the pattern flexible.
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != "load":
        return False
    val = func.value
    if not (isinstance(val, ast.Attribute) and val.attr == "arch"):
        return False
    base = val.value
    if not (isinstance(base, ast.Name) and base.id == "cute"):
        return False
    if len(node.args) != 2:
        return False
    # Second arg must be a VectorType / VectorType.get expression.  Be
    # lax: just require it's a Call so we don't match scalar loads.
    return isinstance(node.args[1], ast.Call)


class _NameRenamer(ast.NodeTransformer):
    """Replace ``ast.Name`` references with a fresh name expression.

    The replacement values are AST subtrees; each substitution returns
    a fresh deep-copy so multiple substitutions stay independent.
    """

    def __init__(self, mapping: dict[str, ast.AST]) -> None:
        self._mapping = mapping

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id in self._mapping:
            replacement = self._mapping[node.id]
            # Always return a fresh deep-copied node so the original
            # remains reusable for repeated lookups.
            return ast.copy_location(_deep_copy_expr(replacement), node)
        return node


def _deep_copy_expr(node: ast.AST) -> ast.expr:
    """Return a deep AST copy of an expression.

    Re-parse via ``ast.unparse`` + ``expr_from_string`` so the result
    uses plain ``ast.AST`` nodes (avoids ``ExtendedAST`` _location
    requirements) and is fully independent of the original tree.
    """
    text = ast.unparse(node)
    result = expr_from_string(text)
    assert isinstance(result, ast.expr)
    return result


class _AssignTargetCollector(ast.NodeVisitor):
    """Collect all simple ``ast.Name`` assignment LHSes in a subtree.

    Used to detect "loop-carried scalar writes": assignments inside the
    inner body whose target name is also defined in the OUTER scope before
    the loop (a reduction accumulator like ``mi``, ``di``).
    """

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.names.add(target.id)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name):
            self.names.add(node.target.id)
        self.generic_visit(node)


def _names_assigned(node: ast.AST) -> set[str]:
    c = _AssignTargetCollector()
    c.visit(node)
    return c.names


def _has_loop_carried_scalar_write(
    inner_body: list[ast.stmt],
    outer_scope_names: set[str],
    rename_groups: dict[str, str] | None = None,
) -> bool:
    """True if any statement in ``inner_body`` writes to a name in
    ``outer_scope_names`` (after canonicalization via ``rename_groups``).

    This indicates the body is a reduction-style loop where the load
    feeds a chain that flows into the next iter via the named
    accumulator.  Pipelining hides that chain's HBM latency.

    Absent any such write (pure stream-store / read-compute-store
    loops), the SASS scheduler can already overlap consecutive iter
    loads naturally and pipelining only inflates register pressure.

    The check is recursive — it traverses the full subtree of each
    inner-body statement so writes nested inside ``range_constexpr``
    V-loops or if-bodies are correctly counted.

    ``rename_groups`` (if supplied) maps pre-rename SSA names (e.g.
    ``v_1_0``) to their post-rename canonical names (e.g. ``mi``).
    The pipeline pass runs BEFORE ``ast_rename`` collapses those
    aliases, so inner writes to ``v_1_0`` need to be matched against
    the outer-scope ``mi`` initialization.
    """
    renames = rename_groups or {}
    # Canonicalize outer_scope_names: anything that's the TARGET of a
    # rename group (i.e., the canonical name) stays as-is; pre-rename
    # aliases get their canonical name added too, so a write to either
    # form counts.
    canonical_outer = set(outer_scope_names)
    for pre, canon in renames.items():
        if pre in outer_scope_names or canon in outer_scope_names:
            canonical_outer.add(pre)
            canonical_outer.add(canon)

    for stmt in inner_body:
        for written in _names_assigned(stmt):
            # Canonicalize the written name through rename_groups.
            canon = renames.get(written, written)
            if written in canonical_outer or canon in canonical_outer:
                return True
    return False


def _try_pipeline_one_outer_loop(
    loop: ast.For,
    constexpr_values: dict[str, int] | None,
    outer_scope_names: set[str] | None = None,
    rename_groups: dict[str, str] | None = None,
) -> list[ast.stmt] | None:
    """Try to software-pipeline a single qualifying outer loop.

    Returns a fresh list ``[prologue_stmt_1, prologue_stmt_2, rewritten_loop]``
    if the loop matches the pattern; otherwise None.

    The rewrite mutates the loop in place (so the returned list contains
    the same outer-loop object with its body modified).
    """
    outer_match = _looks_like_outer_range_loop(loop)
    if outer_match is None:
        return None
    outer_loop, tile_var, start_expr, end_expr, step_expr = outer_match

    # We need a static positive step.  We allow a non-static end_expr.
    step_val = _const_int_value(step_expr, constexpr_values)
    if step_val is None or step_val <= 0:
        return None
    start_val = _const_int_value(start_expr, constexpr_values)
    if start_val is None:
        return None
    end_val = _const_int_value(end_expr, constexpr_values)
    if end_val is None:
        return None
    # Need at least 2 iterations for pipelining to do anything.
    if (end_val - start_val) // step_val < 2:
        return None

    body = outer_loop.body
    if not body or not isinstance(body[0], ast.For):
        return None

    # The inner loop must be ``for lane in range(<int>):`` with EXACTLY
    # one iteration.  When the inner lane loop has REPS > 1, the per-iter
    # prefetch (which advances ``_pipe_lane_base`` to ``tile_var + STEP``
    # but uses the CURRENT lane index) would corrupt the snapshot read
    # by the NEXT lane index in the same outer iter — the next lane
    # snapshot would consume the wrong-outer-iter, wrong-lane prefetch.
    # The common softmax/online-reduction shape has LANE_REPS == 1 so
    # we conservatively skip the rest.
    inner_match = _looks_like_inner_lane_loop(body[0])
    if inner_match is None:
        return None
    inner_loop, lane_var, lane_reps = inner_match
    if lane_reps != 1:
        return None
    inner_body = inner_loop.body
    if len(inner_body) < 2:
        return None

    # First inner statement: ``LANE_BASE = tile_var + ...``.
    lane_base_stmt = inner_body[0]
    lane_base_name = _assignment_lhs_name(lane_base_stmt)
    if lane_base_name is None:
        return None
    assert isinstance(lane_base_stmt, ast.Assign)
    lane_base_rhs = lane_base_stmt.value
    if tile_var not in _names_read(lane_base_rhs):
        return None

    # Second inner statement: ``LOAD_VAR = cute.arch.load(<addr using LANE_BASE>, VEC_TY)``.
    load_stmt = inner_body[1]
    load_var_name = _assignment_lhs_name(load_stmt)
    if load_var_name is None:
        return None
    assert isinstance(load_stmt, ast.Assign)
    load_rhs = load_stmt.value
    if not _is_vec_load_call(load_rhs):
        return None
    if lane_base_name not in _names_read(load_rhs):
        return None

    # We further require that the outer-loop body is JUST the single
    # inner lane loop — no sibling statements at the outer level.  This
    # keeps the rewrite simple (we know exactly where to splice the
    # snapshot / prefetch).
    if len(body) != 1:
        return None

    # P19 optional gate: when ``HELION_LOAD_PIPELINE_CARRIED_ONLY=1`` is
    # set, only pipeline loops whose body has a **loop-carried scalar
    # write** — an assignment to a name defined in the OUTER scope
    # before the loop (a reduction accumulator like ``mi``/``di``).
    #
    # This is the structural marker that the load feeds a per-iter
    # dependency chain, which is exactly when software pipelining
    # most clearly hides HBM latency.  For pure stream-store sweeps
    # (e.g., the consume sweep of two-pass softmax:
    # ``out[k] = exp(values - mi) / di``) there's no inter-iter dep —
    # the SASS scheduler can already overlap consecutive iter loads
    # without help from us, and pipelining only inflates register
    # pressure and instruction count.  In practice the consume-sweep
    # pipeline does sometimes help (the slow ``exp``/``div`` ops give
    # the prefetch room to hide), but on a few softmax shapes where
    # the trip count is awkward (e.g., ``(4096, 12672)`` → 99 trips),
    # NCU shows the consume-sweep pipeline can be a small wall-clock
    # regression.
    #
    # The gate is OPT-IN (default OFF) because in microbench across
    # the standard four shapes (5248/10240/12672/16384) full
    # pipelining (the P18 default) still measures faster at the GPU
    # level than carried-only.  The env var is provided so an
    # autotuner sweep or a future heuristic can flip it without
    # touching code.
    #
    # ``HELION_LOAD_PIPELINE_ALL=1`` is an alias that explicitly
    # disables the gate (pre-P19 behavior — pipeline every qualifying
    # loop).  The two env vars are inverses; the ``ALL`` form is kept
    # for explicitness in test/debug scripts.
    _carried_only = os.environ.get(
        "HELION_LOAD_PIPELINE_CARRIED_ONLY"
    ) and not os.environ.get("HELION_LOAD_PIPELINE_ALL")
    if _carried_only and outer_scope_names is not None:
        if not _has_loop_carried_scalar_write(
            inner_body, outer_scope_names, rename_groups
        ):
            return None

    # Now: allocate pipeline shadow names and build the prologue.
    pipe_lane_base, pipe_load = _new_pipe_names()

    # Build a renaming map.  For the prologue (iter 0) we substitute
    # tile_var = start_expr and lane_var = 0.  In the common case the
    # inner lane loop has range(1) so lane_var=0 is correct.  For
    # lane reps > 1 we still prefetch one outer-loop step ahead; the
    # SASS scheduler can interleave the lane reps within an outer iter.
    prologue_lane_base_rhs = _NameRenamer(
        {tile_var: start_expr, lane_var: ast.Constant(value=0)}
    ).visit(_deep_copy_expr(lane_base_rhs))
    prologue_load_rhs = _NameRenamer(
        {lane_base_name: ast.Name(id=pipe_lane_base, ctx=ast.Load())}
    ).visit(_deep_copy_expr(load_rhs))

    # Emit assignment statements via ``statement_from_string`` so the
    # resulting nodes are plain ``ast.AST`` instances (avoids the
    # ``ExtendedAST`` __init__ which requires a ``_location`` kwarg).
    prologue: list[ast.stmt] = [
        statement_from_string(
            f"{pipe_lane_base} = {ast.unparse(prologue_lane_base_rhs)}"
        ),
        statement_from_string(f"{pipe_load} = {ast.unparse(prologue_load_rhs)}"),
    ]

    # Build the new inner-body prefix:
    #
    #   LANE_BASE = _pipe_lane_base                   # snapshot current
    #   LOAD_VAR  = _pipe_load                        # snapshot current
    #   _pipe_lane_base = <RHS with tile_var -> tile_var + STEP>
    #   _pipe_load      = <load_rhs with lane_base_name -> _pipe_lane_base>
    next_tile_expr = expr_from_string(f"({tile_var} + ({ast.unparse(step_expr)}))")
    next_lane_base_rhs = _NameRenamer({tile_var: next_tile_expr}).visit(
        _deep_copy_expr(lane_base_rhs)
    )
    next_load_rhs = _NameRenamer(
        {lane_base_name: ast.Name(id=pipe_lane_base, ctx=ast.Load())}
    ).visit(_deep_copy_expr(load_rhs))

    snapshot_lane_base = statement_from_string(f"{lane_base_name} = {pipe_lane_base}")
    snapshot_load = statement_from_string(f"{load_var_name} = {pipe_load}")
    prefetch_lane_base = statement_from_string(
        f"{pipe_lane_base} = {ast.unparse(next_lane_base_rhs)}"
    )
    prefetch_load = statement_from_string(f"{pipe_load} = {ast.unparse(next_load_rhs)}")

    new_inner_body: list[ast.stmt] = [
        snapshot_lane_base,
        snapshot_load,
        prefetch_lane_base,
        prefetch_load,
        # Append the rest of the original inner body (after the 2
        # statements we replaced with snapshots).
        *inner_body[2:],
    ]

    inner_loop.body = new_inner_body

    return [*prologue, outer_loop]


def _walk_and_pipeline(
    body: list[ast.stmt],
    constexpr_values: dict[str, int] | None,
    outer_scope_names: set[str] | None = None,
    rename_groups: dict[str, str] | None = None,
) -> list[ast.stmt]:
    """Walk ``body`` recursively, attempting the pipeline transform on
    each top-level qualifying outer loop.  Returns a new body list.

    ``outer_scope_names`` tracks names already defined in scopes that
    enclose ``body``.  Names assigned within ``body`` before each
    candidate loop are unioned in as we walk, so by the time we test a
    given outer loop we know every name that's "live and visible" in
    the enclosing scope of the loop.  See ``_try_pipeline_one_outer_loop``
    for the gating rationale.

    ``rename_groups`` maps pre-rename SSA names to their post-rename
    canonical names so the gate can correctly identify loop-carried
    writes that pre-rename go to alias names (e.g. ``v_1_0 = v_1``
    that ``ast_rename`` will collapse to ``mi = v_1``).
    """
    if outer_scope_names is None:
        outer_scope_names = set()
    # The names visible to each loop body are: enclosing scope names
    # (passed in) plus assignments earlier in this scope.  We track the
    # latter as we walk so each candidate sees its own correct snapshot.
    local_visible: set[str] = set(outer_scope_names)
    new_body: list[ast.stmt] = []
    for stmt in body:
        # Recurse into nested scopes BEFORE the assignment-tracking step
        # so the recursion sees the correct visible set.
        if isinstance(stmt, ast.FunctionDef):
            # A new function scope; do not propagate enclosing names.
            stmt.body = _walk_and_pipeline(
                stmt.body, constexpr_values, rename_groups=rename_groups
            )
            new_body.append(stmt)
            continue
        if isinstance(stmt, (ast.For, ast.If, ast.With)):
            # Recurse into the body first, passing through the current
            # visible set so nested loops can see outer-scope writes.
            stmt.body = _walk_and_pipeline(
                stmt.body, constexpr_values, local_visible, rename_groups
            )
            if isinstance(stmt, (ast.For, ast.If)):
                stmt.orelse = _walk_and_pipeline(
                    stmt.orelse, constexpr_values, local_visible, rename_groups
                )
        if isinstance(stmt, ast.For):
            pipelined = _try_pipeline_one_outer_loop(
                stmt, constexpr_values, local_visible, rename_groups
            )
            if pipelined is not None:
                new_body.extend(pipelined)
                # Update local_visible to include any names written by
                # the rewritten loop's outer-level prologue.
                for ps in pipelined:
                    if isinstance(ps, ast.Assign):
                        for target in ps.targets:
                            if isinstance(target, ast.Name):
                                local_visible.add(target.id)
                continue
        # Track this statement's assignments so subsequent candidate
        # loops in this scope see them as outer-scope-visible.
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    local_visible.add(target.id)
        elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
            local_visible.add(stmt.target.id)
        elif isinstance(stmt, ast.For):
            # The for-loop body may have written to outer-visible names
            # via reduction accumulator updates; union those in too.
            local_visible |= _names_assigned(stmt)
        new_body.append(stmt)
    return new_body


def pipeline_inner_loads(
    body: list[ast.stmt],
    constexpr_values: dict[str, int] | None = None,
    rename_groups: dict[str, str] | None = None,
) -> list[ast.stmt]:
    """Apply the inner-load pipelining peephole to a kernel body.

    ``constexpr_values`` maps module-level constexpr names (e.g.
    ``_BLOCK_SIZE_1``) to their static int value so the range trip
    count can be resolved.

    ``rename_groups`` (optional) maps pre-rename SSA names to their
    post-rename canonical names (e.g. ``{"v_1_0": "mi"}``).  The pass
    runs BEFORE ``ast_rename`` collapses those aliases, so the
    loop-carried-write gate uses this map to identify reduction
    accumulator writes whose pre-rename target is an alias.  Without
    this, the gate would mis-classify the softmax reduce sweep as
    having no loop-carried write and skip pipelining.

    Returns the (possibly modified) body.  Safe to call on any kernel
    body — only rewrites when the strict pattern match succeeds.

    Env vars:
      * ``HELION_DISABLE_LOAD_PIPELINE=1`` — skip the pass entirely.
      * ``HELION_LOAD_PIPELINE_CARRIED_ONLY=1`` — opt in to the P19
        loop-carried-scalar-write gate (only pipeline reduction-style
        loops); see module docstring.
      * ``HELION_LOAD_PIPELINE_ALL=1`` — explicit override that
        force-disables the gate.
    """
    if os.environ.get("HELION_DISABLE_LOAD_PIPELINE"):
        return body
    _PIPE_COUNTER[0] = 0
    return _walk_and_pipeline(body, constexpr_values, rename_groups=rename_groups)
