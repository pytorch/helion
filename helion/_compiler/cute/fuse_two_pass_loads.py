"""AST peephole that fuses redundant loads across two consecutive looped
reductions on the CuTe backend.

Generated reduction kernels (RMS norm, layernorm, softmax, ...) emit two
``for offset in range(...)`` loops that share the same outer iteration and
the same ``ptr.load()`` of the input tensor — once during the reduction
sweep, once again during the post-reduction consume/store sweep. The
second load is pure HBM bandwidth waste because the values already lived
in registers a few statements earlier.

This pass walks the kernel body, finds the pattern::

    for OFFSET in RANGE:
        ...
        NAME1 = (PTR).load() if MASK else CONST   # tracked load
        ...
    ... (post-reduction statements) ...
    for OFFSET in RANGE:                           # same offset, same range
        ...
        NAME2 = (PTR).load() if MASK else CONST   # identical load text
        ...

and rewrites it to cache the load in per-thread scalar registers between
the two loops::

    for OFFSET in RANGE:
        ...
        NAME1 = (PTR).load() if MASK else CONST
        _fuse_cache_<i>[(OFFSET - START) // STEP] = NAME1
        ...
    ... (post-reduction statements) ...
    for OFFSET in RANGE:
        ...
        NAME2 = _fuse_cache_<i>[(OFFSET - START) // STEP]
        ...

The cache is declared once above the first loop. The pass is conservative:
matches by exact unparsed text of the load expression, the mask, and the
two for-loops' ``target`` / ``iter`` / range bounds.
"""

from __future__ import annotations

import ast
import re

from ..ast_extension import statement_from_string


def _range_bounds(node: ast.AST) -> tuple[ast.expr, ast.expr, ast.expr] | None:
    """Extract ``(start, end, step)`` from a Call to ``range(...)``.

    Returns None if the iter is not a 1-, 2-, or 3-arg ``range`` call.
    """
    if not isinstance(node, ast.Call):
        return None
    if not (isinstance(node.func, ast.Name) and node.func.id == "range"):
        return None
    args = node.args
    if not args:
        return None
    if len(args) == 1:
        return (ast.Constant(value=0), args[0], ast.Constant(value=1))
    if len(args) == 2:
        return (args[0], args[1], ast.Constant(value=1))
    if len(args) == 3:
        return (args[0], args[1], args[2])
    return None


def _looks_like_tracked_load(node: ast.AST) -> ast.IfExp | None:
    """Return the IfExp if ``node`` matches the masked-load pattern emitted
    by ``_cute_scalar_load_expr``::

        (PTR).load() if MASK else CONST
    """
    if not isinstance(node, ast.IfExp):
        return None
    # body is the load expression
    body = node.body
    if not isinstance(body, ast.Call):
        return None
    func = body.func
    if not isinstance(func, ast.Attribute) or func.attr != "load":
        return None
    return node


def _looks_like_unmasked_load(node: ast.AST) -> ast.Call | None:
    """Return the Call if ``node`` is an unmasked scalar ``(PTR).load()``.

    The CuTe scalar load emitter drops the ``if mask else CONST`` wrapping
    when no mask is active (e.g. ``weight[:]`` loads in RMS-norm consume
    sweeps); we still want to fuse those across the two passes.
    """
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "load":
            return node
    return None


def _looks_like_vec_load(node: ast.AST) -> ast.Call | None:
    """Return the Call if ``node`` is a ``cute.arch.load(ptr, vec_type)``
    expression — the hoisted U16 vec load emitted by the LoopedReductionStrategy
    ``unroll`` mode.
    """
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if not isinstance(func, ast.Attribute):
        return None
    if func.attr != "load":
        return None
    val = func.value
    # Match either ``cute.arch.load`` or ``cute.arch.<whatever>.load`` — we
    # only care that ``arch`` is in the call chain because that's what the
    # cute helper uses.
    while isinstance(val, ast.Attribute):
        if val.attr == "arch":
            return node
        val = val.value
    return None


def _load_kind(node: ast.AST) -> str | None:
    """Classify the load shape into ``"masked"`` / ``"unmasked"`` / ``"vec"``.

    Returns None when ``node`` doesn't look like a gmem load we can fuse.
    """
    if _looks_like_tracked_load(node) is not None:
        return "masked"
    if _looks_like_vec_load(node) is not None:
        return "vec"
    if _looks_like_unmasked_load(node) is not None:
        return "unmasked"
    return None


def _vec_width(node: ast.AST) -> int | None:
    """Extract the V from ``cute.arch.load(ptr, ir.VectorType.get([V], ...))``.

    Returns None when the expression isn't a recognised vec load.
    """
    if not isinstance(node, ast.Call) or len(node.args) < 2:
        return None
    vec_arg = node.args[1]
    if (
        isinstance(vec_arg, ast.Call)
        and isinstance(vec_arg.func, ast.Attribute)
        and vec_arg.func.attr == "get"
        and len(vec_arg.args) >= 1
    ):
        shape_arg = vec_arg.args[0]
        if isinstance(shape_arg, ast.List) and len(shape_arg.elts) == 1:
            elt = shape_arg.elts[0]
            if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
                return elt.value
    return None


def offset_var_for(loop: ast.For) -> str:
    """Return the for-loop target variable name."""
    assert isinstance(loop.target, ast.Name)
    return loop.target.id


def _dtype_from_default(node: ast.expr) -> str | None:
    """Extract dtype from the else branch of a masked load.

    For ``... if mask else cutlass.Float32(0)``, the else expression is
    ``cutlass.Float32(0)`` and the dtype is ``cutlass.Float32``.
    """
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return ast.unparse(node.func)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _trip_count_for(
    start: ast.expr,
    end: ast.expr,
    step: ast.expr,
    constexpr_values: dict[str, int],
) -> int | None:
    """If start/end/step are constants (possibly wrapped in cutlass.Int32 or
    naming a known constexpr), return the static trip count. Otherwise None.
    """

    def _to_int(expr: ast.expr) -> int | None:
        if isinstance(expr, ast.Constant) and isinstance(expr.value, int):
            return expr.value
        if isinstance(expr, ast.Name) and expr.id in constexpr_values:
            return constexpr_values[expr.id]
        if isinstance(expr, ast.Call) and len(expr.args) == 1:
            inner = expr.args[0]
            if isinstance(inner, ast.Constant) and isinstance(inner.value, int):
                return inner.value
            if isinstance(inner, ast.Name) and inner.id in constexpr_values:
                return constexpr_values[inner.id]
        return None

    s, e, t = _to_int(start), _to_int(end), _to_int(step)
    if s is None or e is None or t is None or t <= 0:
        return None
    if e <= s:
        return 0
    return (e - s + t - 1) // t


def _node_text(node: ast.AST) -> str:
    """Stable text key for matching AST nodes."""
    return ast.unparse(node)


def _rewrite_vec_extract(
    node: ast.AST,
    hoist_var: str,
    cache: str,
    idx_expr: str,
    vec_w: int,
    *,
    use_smem: bool = False,
    thread_count: int = 1,
) -> None:
    """In-place rewrite of ``hoist_var[<vi>]`` -> cache read inside ``node``.

    Used after the vec hoist itself has been deleted from the consume
    sweep so the dependent extracts still resolve.

    Register-cache slot: ``cache[(idx_expr)*V + vi]`` (per-thread).
    SMEM-cache slot: ``cache[((idx_expr)*V + vi) * num_threads + tid]``
    (per-CTA, with the thread offset tail-packed so writes from the
    first sweep land at the same slot the consume sweep reads).
    """

    tid_expr = "cutlass.Int32(cute.arch.thread_idx()[0])"

    class _RewriteVecExtract(ast.NodeTransformer):
        def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
            self.generic_visit(node)
            if isinstance(node.value, ast.Name) and node.value.id == hoist_var:
                vi_text = ast.unparse(node.slice)
                inner = f"({idx_expr}) * {vec_w} + ({vi_text})"
                if use_smem:
                    slot = f"({inner}) * {thread_count} + ({tid_expr})"
                else:
                    slot = inner
                new = ast.parse(f"{cache}[{slot}]").body[0]
                assert isinstance(new, ast.Expr)
                return new.value
            return node

    transformer = _RewriteVecExtract()
    transformer.visit(node)
    ast.fix_missing_locations(node)


def _canonical_load_text(node: ast.AST, lane_var_alias: dict[str, str]) -> str:
    """Stringify a load node with lane-base / hoist variables canonicalised.

    The strategy creates fresh variable names for each sweep
    (``reduction_lane_base_1`` in the reduce sweep, ``reduction_lane_base_2``
    in the consume sweep, ``_unroll_vec_0`` vs ``_unroll_vec_1``...).  The
    fuser matches loads across the two sweeps, so it must compare their
    textual form modulo the rename — otherwise identical loads compare
    unequal and fusion bails.
    """
    text = ast.unparse(node)
    for old, new in lane_var_alias.items():
        # Word-boundary replacement so suffixed vars don't pick up matches
        # for shorter prefixes.
        text = re.sub(rf"\b{re.escape(old)}\b", new, text)
    return text


class _CuteFuseTwoPassLoads(ast.NodeTransformer):
    """See module docstring."""

    def __init__(
        self,
        constexpr_values: dict[str, int] | None = None,
        thread_block_dims: tuple[int, int, int] = (1, 1, 1),
    ) -> None:
        super().__init__()
        self._counter = 0
        self._constexpr_values = constexpr_values or {}
        # Per-axis thread dims for the CUDA thread block.  Used by the
        # SMEM-backed cache path to (a) size the SMEM allocation by
        # total thread count and (b) emit a linear per-thread slot
        # index that is unique across all axes.
        dims = (
            max(1, int(thread_block_dims[0])),
            max(1, int(thread_block_dims[1])),
            max(1, int(thread_block_dims[2])),
        )
        self._thread_block_dims = dims
        self._thread_count = dims[0] * dims[1] * dims[2]

    def _new_cache_name(self) -> str:
        name = f"_fuse_cache_{self._counter}"
        self._counter += 1
        return name

    def _resolve_load_container(
        self,
        outer_loop: ast.For,
        start: ast.expr,
        step: ast.expr,
        trip: int,
    ) -> tuple[list[ast.stmt], str, int] | None:
        """Find the body list holding the actual gmem loads + the index
        expression used to address the cache for one iter of ``outer_loop``.

        Returns ``(container, cache_index_expr, cache_size)`` or None when
        the body shape is not recognised.

        Supported shapes:
          A) Flat body (``for offset ...: load...``) — original simple case.
          B) One nested for-loop (lane loop): ``for lane in range(K)`` —
             cache index is ``(offset - start) // step * K + lane`` and the
             container is the lane loop body.
          C) Two nested for-loops (lane + constexpr vec lane) — the load
             dispatcher hoists vec loads ABOVE the constexpr loop, so the
             container is the lane loop body (between base_stmt and
             vec_for); index is ``(offset - start) // step * K + lane``.
        """
        body = outer_loop.body
        offset_var = offset_var_for(outer_loop)
        cache_index_outer = (
            f"({offset_var} - ({_node_text(start)})) // ({_node_text(step)})"
        )
        for_children = [s for s in body if isinstance(s, ast.For)]
        if not for_children:
            return body, cache_index_outer, trip
        if len(for_children) != 1:
            return None
        # Shape B/C: one nested lane loop
        lane_loop = for_children[0]
        if not isinstance(lane_loop.target, ast.Name):
            return None
        # Resolve lane extent from range(K)
        lane_range = _range_bounds(lane_loop.iter)
        if lane_range is None:
            return None
        lane_start, lane_end, lane_step = lane_range
        lane_trip = _trip_count_for(
            lane_start, lane_end, lane_step, self._constexpr_values
        )
        if lane_trip is None or lane_trip < 1 or lane_trip > 32:
            return None
        lane_var = lane_loop.target.id
        cache_index = f"({cache_index_outer}) * {lane_trip} + ({lane_var})"
        # Check if the lane body contains a NESTED for (the constexpr vec
        # loop in shape C).  If yes, the cache loads sit BEFORE that
        # constexpr loop (the vec-hoist pattern from the LoopedReduction
        # ``unroll`` mode).
        lane_inner_fors = [s for s in lane_loop.body if isinstance(s, ast.For)]
        if not lane_inner_fors:
            # Shape B: lane body itself is the load container.
            return lane_loop.body, cache_index, trip * lane_trip
        if len(lane_inner_fors) != 1:
            return None
        # Shape C: cache the hoisted vec loads (which sit BEFORE the
        # constexpr loop).  The constexpr V-loop itself is left alone
        # because it only reads ``hoist_var[vi].bitcast(...)`` — once the
        # hoist is replaced by a cache read, those inner extracts still
        # work unchanged.
        return lane_loop.body, cache_index, trip * lane_trip

    def _build_second_alias(
        self, first_loop: ast.For, second_loop: ast.For
    ) -> dict[str, str]:
        """Collect per-loop variable renames that need to be normalised so
        the two sweeps' load expressions compare equal.

        Currently handles:
          - LoopedReductionStrategy ``unroll`` mode:
            ``reduction_lane_base_<N>``, ``reduction_vec_lane_<N>``.
          - CuteNDTileStrategy lane / vec naming:
            ``lane_base_<N>``, ``_tile_unroll_vec_<N>_<sweep>``,
            ``vec_lane_<N>``.
        """
        alias: dict[str, str] = {}

        def _collect_assign(loop: ast.For, prefix: str) -> str | None:
            """Return the first variable name assigned inside ``loop``
            whose name starts with ``prefix``.

            We require an assignment (not just a usage) so we always
            pick a single fresh name per sweep, even when downstream
            statements alias the var further.
            """
            for s in ast.walk(loop):
                if isinstance(s, ast.Assign):
                    for t in s.targets:
                        if isinstance(t, ast.Name) and t.id.startswith(prefix):
                            return t.id
            return None

        def _collect_for_target(loop: ast.For, prefix: str) -> str | None:
            """Return the first for-loop target name in ``loop`` starting
            with ``prefix`` (for variables introduced by the constexpr
            ``for vec_lane_<N> in cutlass.range_constexpr(V)`` form).
            """
            for s in ast.walk(loop):
                if (
                    isinstance(s, ast.For)
                    and isinstance(s.target, ast.Name)
                    and s.target.id.startswith(prefix)
                ):
                    return s.target.id
            return None

        for prefix in (
            "reduction_lane_base_",
            "reduction_vec_lane_",
            "lane_base_",
        ):
            first_var = _collect_assign(first_loop, prefix)
            second_var = _collect_assign(second_loop, prefix)
            if first_var and second_var and first_var != second_var:
                # Canonicalise the second sweep's var to the first
                # sweep's name so textual comparison succeeds.
                alias[second_var] = first_var
        # ``vec_lane_<N>`` is a for-loop target, not an assignment.
        for prefix in ("vec_lane_",):
            first_var = _collect_for_target(first_loop, prefix)
            second_var = _collect_for_target(second_loop, prefix)
            if first_var and second_var and first_var != second_var:
                alias[second_var] = first_var
        return alias

    def _dtype_for_load_kind(self, node: ast.AST, kind: str) -> str | None:
        """Extract the storage dtype string for the fragment that backs the
        cached load.  For masked scalar loads, derive from the ``else
        CONST`` branch.  For unmasked / vec loads, derive from the pointer
        or vec-type expression.
        """
        if kind == "masked":
            assert isinstance(node, ast.IfExp)
            return _dtype_from_default(node.orelse)
        if kind == "unmasked":
            # ``(ptr_expr).load()`` — the load expression itself doesn't
            # carry a dtype, but we can look it up later via the assignment
            # target's downstream usage.  For now, fall back to a Float32
            # cache which the cute compiler will coerce.  (RMS-norm consume
            # sweeps in fp16/bf16 take the masked path; this fallback is
            # rarely exercised.)
            return None
        if kind == "vec":
            # ``cute.arch.load(ptr, ir.VectorType.get([V], <elem>.mlir_type))``
            # Pull the element type expression out so the fragment is the
            # right dtype.  The third arg to VectorType.get is the elem.
            assert isinstance(node, ast.Call)
            if len(node.args) >= 2:
                vec_arg = node.args[1]
                # ir.VectorType.get([V], <elem>.mlir_type)
                if (
                    isinstance(vec_arg, ast.Call)
                    and isinstance(vec_arg.func, ast.Attribute)
                    and vec_arg.func.attr == "get"
                    and len(vec_arg.args) >= 2
                ):
                    elem_arg = vec_arg.args[1]
                    if (
                        isinstance(elem_arg, ast.Attribute)
                        and elem_arg.attr == "mlir_type"
                    ):
                        return ast.unparse(elem_arg.value)
            return None
        return None

    def _try_fuse(self, body: list[ast.stmt]) -> list[ast.stmt] | None:
        # Find top-level ``for offset in range(...)`` loops with matching
        # target and range signature -- they don't need to be adjacent in
        # the body. Two-pass reduction kernels typically have
        # post-reduction statements between the two loops.
        loop_indices: list[int] = []
        for i, stmt in enumerate(body):
            if isinstance(stmt, ast.For) and not stmt.orelse:
                loop_indices.append(i)
        if len(loop_indices) < 2:
            return None

        # Group loops by (target, range) signature -- preserve order.
        groups_by_key: dict[tuple[str, str], list[int]] = {}
        for li in loop_indices:
            stmt = body[li]
            assert isinstance(stmt, ast.For)
            if not isinstance(stmt.target, ast.Name):
                continue
            key = (stmt.target.id, _node_text(stmt.iter))
            groups_by_key.setdefault(key, []).append(li)
        groups: list[list[int]] = [g for g in groups_by_key.values() if len(g) >= 2]

        any_fused = False
        new_body = list(body)
        for group in groups:
            if len(group) < 2:
                continue
            first_idx = group[0]
            first_loop = new_body[first_idx]
            assert isinstance(first_loop, ast.For)
            range_args = _range_bounds(first_loop.iter)
            if range_args is None:
                continue
            start, end, step = range_args
            trip = _trip_count_for(start, end, step, self._constexpr_values)
            # Require a static, bounded, non-trivial trip count.  The
            # ``cache_size`` cap is enforced below (and is now SMEM-
            # aware), so we allow large trip counts here — the SMEM
            # backing path covers caches that wouldn't fit in a per-
            # thread register fragment.
            if trip is None or trip <= 1 or trip > 2048:
                continue

            second_idx = group[1]
            second_loop = new_body[second_idx]
            assert isinstance(second_loop, ast.For)
            # Resolve the load-container for each sweep: usually the for
            # body itself, but for wide-chunk / vec configs the loads sit
            # inside a nested lane for-loop.  Track the (load_container,
            # cache_index) pair so the rewrite handles both shapes.
            first_ctx = self._resolve_load_container(first_loop, start, step, trip)
            second_ctx = self._resolve_load_container(second_loop, start, step, trip)
            if first_ctx is None or second_ctx is None:
                continue
            first_container, first_cache_index, first_cache_size = first_ctx
            second_container, second_cache_index, second_cache_size = second_ctx
            if first_cache_size != second_cache_size:
                continue
            cache_size = first_cache_size
            # Default policy: register-backed cache only when
            # ``cache_size <= 64``; otherwise skip fusion.
            #
            # Why skip beyond 64 instead of switching to SMEM:
            # empirically (see task P10), the SMEM-backed cache eats
            # 25+ KB of per-CTA SMEM for the autotuner-picked
            # warp-reduction softmax config (cache_size=99, V=4,
            # threads=32), which crashes occupancy (~14% on B200).  In
            # parallel, the SMEM-write + barrier + SMEM-read overhead
            # outweighs the second-pass gmem read that the kernel's
            # natural L1 reuse mostly serves already.  The autotuner
            # accordingly steers to fusion-friendly configs (wider
            # block + more threads, cache_size=13) when fusion is
            # available, which still leaves the warp-reduction shape
            # ~13% faster overall.
            #
            # Env-var escape hatches for experimentation:
            #   HELION_FUSER_MODE=disabled — never fuse.
            #   HELION_FUSER_MODE=register — force register fragment up
            #     to cache_size=1024 (will spill / drop occupancy for
            #     large caches).
            #   HELION_FUSER_MODE=smem — force SMEM-backed cache up to
            #     cache_size=1024.  Requires ``thread_count`` to be
            #     plumbed correctly from the dispatch layer.
            import os

            _fuser_mode = os.environ.get("HELION_FUSER_MODE", "auto")
            if _fuser_mode == "disabled":
                continue
            if _fuser_mode == "register":
                use_smem = False
                if cache_size > 1024:
                    continue
            elif _fuser_mode == "smem":
                use_smem = True
                if cache_size > 1024:
                    continue
            else:  # auto
                if cache_size > 64:
                    continue
                use_smem = False
            cache_index = first_cache_index
            second_cache_index_str = second_cache_index

            # Build a canonical-form alias map that smooths over per-sweep
            # variable renames (``reduction_lane_base_1`` vs
            # ``reduction_lane_base_2``, lane var counters, etc.) so the
            # two sweeps' loads compare equal by text.
            second_alias = self._build_second_alias(first_loop, second_loop)

            # Collect tracked loads from the first container.  Vec loads
            # (the ``unroll`` mode's U16 vec hoist) cache via a *scalar*
            # fragment of ``cache_size * V`` slots — one slot per
            # extracted lane — because the CUTLASS DSL's
            # ``cute.make_fragment(N, dtype)`` does not currently accept a
            # vec element type.  Scalar (masked / unmasked) loads cache as
            # the existing fast path.
            tracked: dict[str, tuple[int, str, str, str | None, int | None]] = {}
            for j, s in enumerate(first_container):
                if (
                    isinstance(s, ast.Assign)
                    and len(s.targets) == 1
                    and isinstance(s.targets[0], ast.Name)
                ):
                    kind = _load_kind(s.value)
                    if kind is None:
                        continue
                    dtype = self._dtype_for_load_kind(s.value, kind)
                    if dtype is None:
                        continue
                    v_width = _vec_width(s.value) if kind == "vec" else 1
                    if kind == "vec" and v_width is None:
                        continue
                    tracked[_node_text(s.value)] = (
                        j,
                        s.targets[0].id,
                        kind,
                        dtype,
                        v_width,
                    )
            if not tracked:
                continue

            # Match loads in the second container.
            assignments_to_rewrite: list[tuple[int, str, str]] = []
            for j, s in enumerate(second_container):
                if (
                    isinstance(s, ast.Assign)
                    and len(s.targets) == 1
                    and isinstance(s.targets[0], ast.Name)
                ):
                    kind = _load_kind(s.value)
                    if kind is None:
                        continue
                    # Canonicalise the second sweep's load text against the
                    # first sweep's variable names before keying.
                    key = _canonical_load_text(s.value, second_alias)
                    if key in tracked:
                        assignments_to_rewrite.append((j, s.targets[0].id, key))
            if not assignments_to_rewrite:
                continue

            # Build cache declarations.  For vec loads, allocate
            # ``cache_size * V`` scalar slots; for scalar loads use the
            # original ``cache_size`` count.
            #
            # Backing:
            #   - Register fragment for ``cache_size <= 64``: lowest
            #     latency, no sync needed, fits in registers.
            #   - SMEM tensor for larger caches: allocated once at the
            #     top, indexed per-thread.  Sync inserted between the
            #     two sweeps so the consume read sees populated slots.
            cache_names: dict[str, tuple[str, int]] = {}
            cache_decls: list[ast.stmt] = []
            # Build the linear per-thread index expression covering all
            # populated thread-block axes (axis 0 = warp lanes, axis 1
            # = additional thread rows, axis 2 = z).  For a 1-D thread
            # block this is just thread_idx[0].  For a 2-D block (used
            # by per-row layernorm-style kernels) this picks up
            # thread_idx[1] * dim0 so each row gets its own SMEM
            # region.
            dim0, dim1, dim2 = self._thread_block_dims
            tid_terms = ["cutlass.Int32(cute.arch.thread_idx()[0])"]
            if dim1 > 1:
                tid_terms.append(f"cutlass.Int32(cute.arch.thread_idx()[1]) * {dim0}")
            if dim2 > 1:
                tid_terms.append(
                    f"cutlass.Int32(cute.arch.thread_idx()[2]) * {dim0 * dim1}"
                )
            tid_expr = " + ".join(tid_terms)
            for key, (_j, _name, _kind, dtype, vec_w) in tracked.items():
                if not any(akey == key for _, _, akey in assignments_to_rewrite):
                    continue
                if dtype is None or vec_w is None:
                    continue
                cache = self._new_cache_name()
                cache_names[key] = (cache, vec_w)
                cache_total_per_thread = cache_size * vec_w
                if use_smem:
                    cache_total = cache_total_per_thread * self._thread_count
                    smem_ptr = f"{cache}_ptr"
                    cache_decls.extend(
                        [
                            statement_from_string(
                                f"{smem_ptr} = cute.arch.alloc_smem({dtype}, {cache_total})"
                            ),
                            statement_from_string(
                                f"{cache} = cute.make_tensor({smem_ptr}, ({cache_total},))"
                            ),
                        ]
                    )
                else:
                    cache_total = cache_total_per_thread
                    cache_decls.append(
                        statement_from_string(
                            f"{cache} = cute.make_fragment({cache_total}, {dtype})"
                        )
                    )
            if not cache_names:
                continue

            # Helper: build the cache slot expression for a given outer
            # index expression and vec lane.
            #
            # SMEM cache layout: per-thread contiguous region —
            #   slot = tid * (cache_size * vec_w) + iter * vec_w + vi
            # This keeps each thread's slots clustered in SMEM (so the
            # compiler can issue vector stores/loads), and avoids bank
            # conflicts as long as the per-thread stride is not a
            # multiple of 32 * 4 bytes.
            #
            # For register fragments we keep the original per-thread
            # layout: slot = iter_idx * vec_w + vi.
            #
            # Capture loop-local ``cache_size`` / ``tid_expr`` /
            # ``use_smem`` via default args to satisfy ruff B023.
            def _slot_expr(
                idx_text: str,
                vec_w_: int,
                vi_text: str,
                _cache_size: int = cache_size,
                _tid_expr: str = tid_expr,
                _use_smem: bool = use_smem,
            ) -> str:
                if vec_w_ == 1:
                    inner = f"({idx_text})"
                else:
                    inner = f"({idx_text}) * {vec_w_} + ({vi_text})"
                if _use_smem:
                    return f"({_tid_expr}) * {_cache_size * vec_w_} + ({inner})"
                return inner

            # Rewrite the first container: append cache writes after each
            # tracked load assignment.  Vec loads write V scalars per
            # cache slot; scalar loads write a single scalar.
            new_first_body: list[ast.stmt] = []
            for s in first_container:
                new_first_body.append(s)
                if (
                    isinstance(s, ast.Assign)
                    and len(s.targets) == 1
                    and isinstance(s.targets[0], ast.Name)
                ):
                    if _load_kind(s.value) is None:
                        continue
                    key = _node_text(s.value)
                    entry = cache_names.get(key)
                    if entry is None:
                        continue
                    cache, vec_w = entry
                    name = s.targets[0].id
                    if vec_w == 1:
                        slot = _slot_expr(cache_index, 1, "0")
                        new_first_body.append(
                            statement_from_string(f"{cache}[{slot}] = {name}")
                        )
                    else:
                        for v in range(vec_w):
                            slot = _slot_expr(cache_index, vec_w, str(v))
                            new_first_body.append(
                                statement_from_string(f"{cache}[{slot}] = {name}[{v}]")
                            )
            first_container[:] = new_first_body

            # Rewrite the second container: replace each matched load.
            # Scalar loads become a single cache read.  Vec loads are
            # eliminated entirely (the consume sweep's hoist disappears)
            # and any downstream ``hoist_var[vi]`` extracts inside the
            # nested constexpr V-loop are rewritten to read from the cache
            # at the appropriate slot expression.
            vec_extract_rewrites: list[tuple[str, str, str, int, bool, int]] = []
            new_second_body: list[ast.stmt] = []
            for s in second_container:
                if (
                    isinstance(s, ast.Assign)
                    and len(s.targets) == 1
                    and isinstance(s.targets[0], ast.Name)
                ):
                    kind = _load_kind(s.value)
                    if kind is not None:
                        key = _canonical_load_text(s.value, second_alias)
                        entry = cache_names.get(key)
                        if entry is not None:
                            cache, vec_w = entry
                            name = s.targets[0].id
                            if vec_w == 1:
                                slot = _slot_expr(second_cache_index_str, 1, "0")
                                new_second_body.append(
                                    statement_from_string(f"{name} = {cache}[{slot}]")
                                )
                            else:
                                # Drop the hoist entirely; remember that
                                # ``name[vi]`` needs to be rewritten to a
                                # cache read in subsequent statements
                                # (especially inside the constexpr V-loop).
                                vec_extract_rewrites.append(
                                    (
                                        name,
                                        cache,
                                        second_cache_index_str,
                                        vec_w,
                                        use_smem,
                                        self._thread_count,
                                    )
                                )
                            continue
                new_second_body.append(s)
            # Apply the vec extract rewrites recursively to ``new_second_body``.
            if vec_extract_rewrites:
                for (
                    hoist_var,
                    cache,
                    idx_expr,
                    vec_w,
                    use_smem_,
                    tcount,
                ) in vec_extract_rewrites:
                    for stmt in new_second_body:
                        _rewrite_vec_extract(
                            stmt,
                            hoist_var,
                            cache,
                            idx_expr,
                            vec_w,
                            use_smem=use_smem_,
                            thread_count=tcount,
                        )
            second_container[:] = new_second_body

            # Insert cache declarations before the first loop, in
            # original order (so ``alloc_smem`` precedes
            # ``make_tensor``).  Insert at incrementing positions
            # rather than reusing ``first_idx`` (which would reverse
            # them).
            for offset, decl in enumerate(cache_decls):
                new_body.insert(first_idx + offset, decl)
                # Adjust the second-loop index forward.
                second_idx += 1
            if use_smem:
                # SMEM-backed cache requires a CTA-wide barrier after the
                # first sweep populates the cache and before the second
                # sweep reads it back. The two-pass kernels' inter-sweep
                # code (computing the row-wise reduction result, etc.)
                # may write to ``mi``/``di`` registers but doesn't touch
                # SMEM, so one barrier just before the consume loop is
                # sufficient.
                new_body.insert(
                    second_idx,
                    statement_from_string("cute.arch.sync_threads()"),
                )
                second_idx += 1
            any_fused = True

        if not any_fused:
            return None
        return new_body

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        new_body = self._try_fuse(node.body)
        if new_body is not None:
            node.body = new_body
        # Recurse for nested functions (unlikely in our kernels).
        return self.generic_visit(node)  # type: ignore[return-value]


def fuse_two_pass_loads(
    body: list[ast.stmt],
    constexpr_values: dict[str, int] | None = None,
    *,
    thread_block_dims: tuple[int, int, int] = (1, 1, 1),
) -> list[ast.stmt]:
    """Apply two-pass load fusion to a list of statements (the device kernel
    body). Returns the (possibly modified) body.

    ``constexpr_values`` maps constexpr name -> static integer value so
    the pass can resolve ``range(..., step=cutlass.Int32(NAME))`` trip
    counts when NAME is inlined as a kernel-level constexpr.

    ``thread_block_dims`` is the launch-time thread-block shape
    ``(x, y, z)``.  Used to (a) decide between register-backed cache
    (small caches, occupancy-friendly) and SMEM-backed cache (large
    caches), and (b) build a per-thread linear slot index for the SMEM
    path so kernels with 2-D / 3-D thread blocks (layernorm,
    block-pointer softmax) don't have rows clobber each other's slots.

    Safe to call on any kernel body — only rewrites when a strict pattern
    match succeeds.
    """
    transformer = _CuteFuseTwoPassLoads(
        constexpr_values=constexpr_values,
        thread_block_dims=thread_block_dims,
    )
    new_body = transformer._try_fuse(body)
    if new_body is None:
        return body
    return new_body
