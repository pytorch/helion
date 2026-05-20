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


class _CuteFuseTwoPassLoads(ast.NodeTransformer):
    """See module docstring."""

    def __init__(self, constexpr_values: dict[str, int] | None = None) -> None:
        super().__init__()
        self._counter = 0
        self._constexpr_values = constexpr_values or {}

    def _new_cache_name(self) -> str:
        name = f"_fuse_cache_{self._counter}"
        self._counter += 1
        return name

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
            # Require a static, bounded, non-trivial trip count. Keep the
            # cap conservative to avoid runaway register pressure: 32 fp32
            # per thread (= 128 B) is plenty for RMS-norm-style kernels at
            # the chunk sizes the autotuner picks. Anything larger
            # currently means we're at a regime where TMA/cute.copy is the
            # right fix anyway.
            if trip is None or trip <= 1 or trip > 32:
                continue

            second_idx = group[1]
            second_loop = new_body[second_idx]
            assert isinstance(second_loop, ast.For)
            # Only the simple (no nested lane loop) case is fused. With a
            # nested lane loop the cache index mixes runtime ``offset``
            # with the lane variable, and CUTLASS DSL falls back to local
            # memory (spilled to stack), which regresses performance.
            container_first = first_loop.body
            container_second = second_loop.body
            cache_size = trip
            cache_index = (
                f"({offset_var_for(first_loop)} - ({_node_text(start)})) // "
                f"({_node_text(step)})"
            )
            # Bail out on nested for-loops in either body to keep the
            # simple-case invariant.
            if any(isinstance(s, ast.For) for s in container_first):
                continue
            if any(isinstance(s, ast.For) for s in container_second):
                continue

            # Collect tracked loads, keyed by unparsed text.
            tracked: dict[str, tuple[int, str]] = {}
            for j, s in enumerate(container_first):
                if (
                    isinstance(s, ast.Assign)
                    and len(s.targets) == 1
                    and isinstance(s.targets[0], ast.Name)
                ):
                    load = _looks_like_tracked_load(s.value)
                    if load is not None:
                        tracked[_node_text(load)] = (j, s.targets[0].id)
            if not tracked:
                continue

            # Find matching loads in the second loop's container.
            assignments_to_rewrite: list[tuple[int, str, str]] = []
            for j, s in enumerate(container_second):
                if (
                    isinstance(s, ast.Assign)
                    and len(s.targets) == 1
                    and isinstance(s.targets[0], ast.Name)
                ):
                    load = _looks_like_tracked_load(s.value)
                    if load is None:
                        continue
                    key = _node_text(load)
                    if key in tracked:
                        assignments_to_rewrite.append((j, s.targets[0].id, key))
            if not assignments_to_rewrite:
                continue

            # Build cache declarations.
            cache_names: dict[str, str] = {}
            cache_decls: list[ast.stmt] = []
            for key, (j, _name) in tracked.items():
                if not any(akey == key for _, _, akey in assignments_to_rewrite):
                    continue
                load_stmt = container_first[j]
                assert isinstance(load_stmt, ast.Assign)
                load_ifexp = _looks_like_tracked_load(load_stmt.value)
                assert load_ifexp is not None
                dtype = _dtype_from_default(load_ifexp.orelse)
                if dtype is None:
                    continue
                cache = self._new_cache_name()
                cache_names[key] = cache
                cache_decls.append(
                    statement_from_string(
                        f"{cache} = cute.make_fragment({cache_size}, {dtype})"
                    )
                )
            if not cache_names:
                continue

            # Rewrite the first loop: insert cache writes after each tracked
            # load.
            new_first_body: list[ast.stmt] = []
            for s in container_first:
                new_first_body.append(s)
                if (
                    isinstance(s, ast.Assign)
                    and len(s.targets) == 1
                    and isinstance(s.targets[0], ast.Name)
                ):
                    load = _looks_like_tracked_load(s.value)
                    if load is None:
                        continue
                    key = _node_text(load)
                    cache = cache_names.get(key)
                    if cache is None:
                        continue
                    name = s.targets[0].id
                    new_first_body.append(
                        statement_from_string(f"{cache}[{cache_index}] = {name}")
                    )
            first_loop.body = new_first_body

            # Rewrite the second loop: replace each matched load with a
            # read from the cache.
            new_second_body: list[ast.stmt] = []
            for s in container_second:
                if (
                    isinstance(s, ast.Assign)
                    and len(s.targets) == 1
                    and isinstance(s.targets[0], ast.Name)
                ):
                    load = _looks_like_tracked_load(s.value)
                    if load is not None:
                        key = _node_text(load)
                        cache = cache_names.get(key)
                        if cache is not None:
                            name = s.targets[0].id
                            new_second_body.append(
                                statement_from_string(
                                    f"{name} = {cache}[{cache_index}]"
                                )
                            )
                            continue
                new_second_body.append(s)
            second_loop.body = new_second_body

            # Insert cache declarations before the first loop.
            for decl in cache_decls:
                new_body.insert(first_idx, decl)
                # Adjust subsequent indices in this group.
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
) -> list[ast.stmt]:
    """Apply two-pass load fusion to a list of statements (the device kernel
    body). Returns the (possibly modified) body.

    ``constexpr_values`` maps constexpr name -> static integer value so
    the pass can resolve ``range(..., step=cutlass.Int32(NAME))`` trip
    counts when NAME is inlined as a kernel-level constexpr.

    Safe to call on any kernel body — only rewrites when a strict pattern
    match succeeds.
    """
    transformer = _CuteFuseTwoPassLoads(constexpr_values=constexpr_values)
    new_body = transformer._try_fuse(body)
    if new_body is None:
        return body
    return new_body
