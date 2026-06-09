"""AST pre-pass that rewrites the online two-pass softmax pattern into the
equivalent 3-pass form for the CuTe backend.

Shape gating: the rewrite only fires when the reduction-axis extent ``N``
exceeds ``HELION_ONLINE_TO_3PASS_MIN_N`` (default 2048).  The 3-pass form
adds an extra inner sweep and so introduces additional SMEM-reduction
sync per outer iter; for small ``N`` the per-sweep fixed cost dominates
and the rewrite REGRESSES wall-clock.  For large ``N`` (where the online
merge's rescale dominates) the independent reductions in the 3-pass form
win materially.  The cutoff was picked from B200 microbench on
``softmax_two_pass`` shapes (4096, 256/6400/12672/16384):
``N >= 2048`` flips the sign of the perf delta.


The CuTe backend's two-pass softmax kernel (``examples/softmax.py::softmax_two_pass``)
expresses the algorithm as one outer ``for tile_m`` loop with TWO inner
``for tile_n`` loops:

    for tile_m in hl.tile(m, block_size=block_size_m):
        mi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        di = hl.zeros([tile_m], dtype=torch.float32)
        # PASS 1 (online merge): running max + running sum with rescale
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            local_amax = torch.amax(values, dim=1)
            mi_next = torch.maximum(mi, local_amax)
            di = di * torch.exp(mi - mi_next) + torch.exp(
                values - mi_next[:, None]
            ).sum(dim=1)
            mi = mi_next
        # PASS 2 (consume): normalize using final mi/di
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            out[tile_m, tile_n] = torch.exp(values - mi[:, None]) / di[:, None]

The online merge inside pass 1 has TWO sequential SMEM reductions per outer
iter (the amax warp/shared reduce and the exp-sum reduce), and the rescale
introduces a data dependency between them.  In CuTe codegen this compiles
to a kernel that hits a structural perf ceiling (~1750 GB/s on B200 for
(4096, 12672) fp16).

The 3-pass form computes the SAME final ``mi``/``di`` because:
  * max-pass final mi = max over all tiles of local_amax — equivalent to
    the running maximum once the loop has visited every tile.
  * sum-pass final di = sum over all tiles of sum(exp(values - mi_final))
    — equivalent to the running rescaled sum once mi has reached its
    final value.

Rewriting the first inner loop into TWO separate sweeps (max-only, then
sum-only) lets each reduction stand on its own and compiles to materially
faster code (the structural ceiling goes away because the two reductions
no longer share an outer iter).  Numerical roundoff differs by a factor
that's smaller than the fp16 tolerance used for softmax correctness
(verified end-to-end against ``torch.nn.functional.softmax``).

This pass runs on the user's source AST BEFORE tracing, gated on the CuTe
backend.  Detection is intentionally conservative — only matches the
exact ``softmax_two_pass`` body shape — so it won't fire on lookalike
kernels.  Set ``HELION_DISABLE_ONLINE_TO_3PASS=1`` to skip it (for A/B
testing or if a downstream variant breaks).
"""

from __future__ import annotations

import ast
import os
from typing import TYPE_CHECKING

from ..ast_extension import ExtendedAST
from ..ast_extension import create

if TYPE_CHECKING:
    from ..host_function import HostFunction


def _is_hl_call(node: ast.AST, attr: str) -> bool:
    """Return True if ``node`` is a Call to ``hl.<attr>(...)``."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == attr
        and isinstance(func.value, ast.Name)
        and func.value.id == "hl"
    )


def _is_torch_call(node: ast.AST, attr: str) -> bool:
    """Return True if ``node`` is a Call to ``torch.<attr>(...)``."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == attr
        and isinstance(func.value, ast.Name)
        and func.value.id == "torch"
    )


def _name(node: ast.AST | None) -> str | None:
    """Return the identifier if ``node`` is a single ast.Name, else None."""
    if isinstance(node, ast.Name):
        return node.id
    return None


def _single_target_name(assign: ast.AST) -> str | None:
    """Return the target identifier of a single-target Assign, else None."""
    if not isinstance(assign, ast.Assign) or len(assign.targets) != 1:
        return None
    return _name(assign.targets[0])


def _is_tile_call(node: ast.AST) -> bool:
    """Detect ``hl.tile(<extent>, ...)`` calls used as for-loop iterators."""
    return _is_hl_call(node, "tile")


def _expr_unparse(node: ast.AST) -> str:
    """Stable canonical unparse for an expression."""
    return ast.unparse(node)


def _strip_doc(stmts: list[ast.stmt]) -> list[ast.stmt]:
    """Drop a leading docstring Expr if present (it's metadata, not code)."""
    if (
        stmts
        and isinstance(stmts[0], ast.Expr)
        and isinstance(stmts[0].value, ast.Constant)
        and isinstance(stmts[0].value.value, str)
    ):
        return stmts[1:]
    return stmts


def _ensure_for_extended(stmt: ast.AST) -> ast.For | None:
    """Return ``stmt`` as an ast.For if it is one, else None."""
    if isinstance(stmt, ast.For):
        return stmt
    return None


def _ext_copy(node: ast.AST) -> ast.AST:
    """Deepcopy an AST node, preserving ExtendedAST mixin attributes when
    present so source locations / loop type tags survive the rewrite.

    Standard ``copy.deepcopy`` does not work on ``ExtendedAST`` because
    its ``__init__`` requires the keyword-only ``_location`` argument
    (the default deepcopy reconstructor uses positional args).  Walk the
    tree manually instead, recreating each ExtendedAST node via its
    ``copy()`` helper.
    """
    if isinstance(node, list):
        # pyrefly: ignore [bad-return]
        return [_ext_copy(x) for x in node]  # type: ignore[return-value]
    if not isinstance(node, ast.AST):
        return node
    if isinstance(node, ExtendedAST):
        new_fields = {field: _ext_copy(getattr(node, field)) for field in node._fields}
        # pyrefly: ignore [bad-return]
        return node.copy(**new_fields)
    # Plain ast.AST (no ExtendedAST mixin) — recreate via class.
    cls = type(node)
    new_fields = {
        field: _ext_copy(getattr(node, field))
        for field in node._fields
        if hasattr(node, field)
    }
    new_node = cls(**new_fields)
    for attr in getattr(node, "_attributes", ()):
        if hasattr(node, attr):
            setattr(new_node, attr, getattr(node, attr))
    return new_node


def _make_for_loop(template_loop: ast.For, body: list[ast.stmt]) -> ast.For:
    """Build a new ``for`` loop that reuses ``template_loop``'s target and
    iter (so detection key + downstream tile binding stay identical) but
    has the given ``body``.
    """
    target = _ext_copy(template_loop.target)
    iter_node = _ext_copy(template_loop.iter)
    if isinstance(template_loop, ExtendedAST):
        return template_loop.copy(  # pyrefly: ignore [bad-return]
            target=target,
            iter=iter_node,
            body=body,
            orelse=[],
        )
    return create(
        ast.For,
        target=target,
        iter=iter_node,
        body=body,
        orelse=[],
        type_comment=None,
    )


def _detect_online_softmax(
    outer_body: list[ast.stmt],
) -> tuple[int, dict[str, object]] | None:
    """Detect the online-softmax pattern in an outer ``for tile_m`` loop body.

    Returns ``(start_index, info)`` where ``start_index`` is the position of
    the first statement (the ``mi = hl.full(...)``) and ``info`` carries the
    matched names + AST nodes needed by the rewriter.  Returns None if the
    pattern doesn't match.

    The detector finds the FIRST occurrence of the 4-statement sequence:

        mi   = hl.full([tile_m], float('-inf'), dtype=torch.float32)
        di   = hl.zeros([tile_m], dtype=torch.float32)  (or hl.full(..., 0))
        for tile_n in hl.tile(...):
            <online merge body>
        for tile_n in hl.tile(...):
            <consume body>

    where the two inner loops share the same iter expression text and the
    online-merge body matches the exact 5-statement shape used by
    ``examples/softmax.py::softmax_two_pass``.
    """
    n = len(outer_body)
    for i in range(n - 3):
        # Look for the (mi init, di init, first inner for, second inner for)
        # contiguous slice.  We allow other statements before/after but the
        # 4-statement match must be contiguous so the rewrite can splice
        # safely.
        s_mi = outer_body[i]
        s_di = outer_body[i + 1]
        s_loop1 = outer_body[i + 2]
        s_loop2 = outer_body[i + 3]

        mi_name = _single_target_name(s_mi)
        di_name = _single_target_name(s_di)
        if mi_name is None or di_name is None:
            continue
        assert isinstance(s_mi, ast.Assign)
        assert isinstance(s_di, ast.Assign)

        # mi = hl.full([tile_m], float('-inf'), dtype=torch.float32)
        if not _is_hl_call(s_mi.value, "full"):
            continue
        mi_call = s_mi.value
        assert isinstance(mi_call, ast.Call)
        if len(mi_call.args) < 2:
            continue
        mi_shape = mi_call.args[0]
        if not (
            isinstance(mi_shape, ast.List)
            and len(mi_shape.elts) == 1
            and isinstance(mi_shape.elts[0], ast.Name)
        ):
            continue
        tile_m_name = mi_shape.elts[0].id
        # The init value must be -inf (float('-inf') or a Constant("-inf"))
        init_val = mi_call.args[1]
        init_text = _expr_unparse(init_val)
        if init_text not in {"float('-inf')", 'float("-inf")', "-math.inf"}:
            continue

        # di = hl.zeros([tile_m], dtype=torch.float32)  -OR-
        # di = hl.full([tile_m], 0, dtype=torch.float32)
        if _is_hl_call(s_di.value, "zeros"):
            di_call = s_di.value
            assert isinstance(di_call, ast.Call)
            if not di_call.args:
                continue
            di_shape = di_call.args[0]
            if not (
                isinstance(di_shape, ast.List)
                and len(di_shape.elts) == 1
                and _name(di_shape.elts[0]) == tile_m_name
            ):
                continue
            di_init_value_repr = "0.0"  # canonical
        elif _is_hl_call(s_di.value, "full"):
            di_call = s_di.value
            assert isinstance(di_call, ast.Call)
            if len(di_call.args) < 2:
                continue
            di_shape = di_call.args[0]
            if not (
                isinstance(di_shape, ast.List)
                and len(di_shape.elts) == 1
                and _name(di_shape.elts[0]) == tile_m_name
            ):
                continue
            di_init_text = _expr_unparse(di_call.args[1])
            if di_init_text not in {"0", "0.0"}:
                continue
            di_init_value_repr = "0.0"
        else:
            continue

        # First inner for-loop: online merge
        loop1 = _ensure_for_extended(s_loop1)
        loop2 = _ensure_for_extended(s_loop2)
        if loop1 is None or loop2 is None:
            continue
        if not _is_tile_call(loop1.iter) or not _is_tile_call(loop2.iter):
            continue
        if _expr_unparse(loop1.iter) != _expr_unparse(loop2.iter):
            continue
        # Same target tile_n name in both loops.
        tile_n1 = _name(loop1.target)
        tile_n2 = _name(loop2.target)
        if tile_n1 is None or tile_n2 is None or tile_n1 != tile_n2:
            continue
        tile_n_name = tile_n1

        # Match the online-merge body shape exactly:
        #   values     = x[tile_m, tile_n]
        #   local_amax = torch.amax(values, dim=1)
        #   mi_next    = torch.maximum(mi, local_amax)
        #   di = di * torch.exp(mi - mi_next) + torch.exp(
        #       values - mi_next[:, None]
        #   ).sum(dim=1)
        #   mi = mi_next
        body1 = _strip_doc(list(loop1.body))
        if len(body1) != 5:
            continue
        b0, b1, b2, b3, b4 = body1
        values_name = _single_target_name(b0)
        if values_name is None:
            continue
        assert isinstance(b0, ast.Assign)
        # x[tile_m, tile_n] — capture the source-tensor name to reuse in
        # the rewritten loops, but accept any name (we only need to ensure
        # it's a 2-D subscript over (tile_m, tile_n)).
        if not isinstance(b0.value, ast.Subscript):
            continue
        src_name = _name(b0.value.value)
        if src_name is None:
            continue
        sl = b0.value.slice
        if not (isinstance(sl, ast.Tuple) and len(sl.elts) == 2):
            continue
        if _name(sl.elts[0]) != tile_m_name or _name(sl.elts[1]) != tile_n_name:
            continue

        # local_amax = torch.amax(values, dim=1)
        local_amax_name = _single_target_name(b1)
        if local_amax_name is None:
            continue
        assert isinstance(b1, ast.Assign)
        if not _is_torch_call(b1.value, "amax"):
            continue
        amax_call = b1.value
        assert isinstance(amax_call, ast.Call)
        if not (amax_call.args and _name(amax_call.args[0]) == values_name):
            continue
        amax_dim = None
        for kw in amax_call.keywords:
            if kw.arg == "dim":
                amax_dim = _expr_unparse(kw.value)
        # also support positional dim
        if amax_dim is None and len(amax_call.args) >= 2:
            amax_dim = _expr_unparse(amax_call.args[1])
        if amax_dim != "1":
            continue

        # mi_next = torch.maximum(mi, local_amax)
        mi_next_name = _single_target_name(b2)
        if mi_next_name is None:
            continue
        assert isinstance(b2, ast.Assign)
        if not _is_torch_call(b2.value, "maximum"):
            continue
        mx_call = b2.value
        assert isinstance(mx_call, ast.Call)
        if len(mx_call.args) != 2:
            continue
        if (
            _name(mx_call.args[0]) != mi_name
            or _name(mx_call.args[1]) != local_amax_name
        ):
            continue

        # di = di * torch.exp(mi - mi_next) + torch.exp(
        #     values - mi_next[:, None]
        # ).sum(dim=1)
        di_update_name = _single_target_name(b3)
        if di_update_name != di_name:
            continue
        assert isinstance(b3, ast.Assign)
        if not isinstance(b3.value, ast.BinOp) or not isinstance(b3.value.op, ast.Add):
            continue
        left = b3.value.left
        right = b3.value.right
        # left: di * torch.exp(mi - mi_next)
        if not (isinstance(left, ast.BinOp) and isinstance(left.op, ast.Mult)):
            continue
        if _name(left.left) != di_name:
            continue
        if not _is_torch_call(left.right, "exp"):
            continue
        exp_call = left.right
        assert isinstance(exp_call, ast.Call)
        if len(exp_call.args) != 1:
            continue
        sub_expr = exp_call.args[0]
        if not (isinstance(sub_expr, ast.BinOp) and isinstance(sub_expr.op, ast.Sub)):
            continue
        if _name(sub_expr.left) != mi_name or _name(sub_expr.right) != mi_next_name:
            continue
        # right: torch.exp(values - mi_next[:, None]).sum(dim=1)
        if not (
            isinstance(right, ast.Call)
            and isinstance(right.func, ast.Attribute)
            and right.func.attr == "sum"
        ):
            continue
        sum_kw_dim = None
        for kw in right.keywords:
            if kw.arg == "dim":
                sum_kw_dim = _expr_unparse(kw.value)
        if sum_kw_dim is None and len(right.args) >= 1:
            sum_kw_dim = _expr_unparse(right.args[0])
        if sum_kw_dim != "1":
            continue
        sum_target = right.func.value
        if not _is_torch_call(sum_target, "exp"):
            continue
        sum_exp_call = sum_target
        assert isinstance(sum_exp_call, ast.Call)
        if len(sum_exp_call.args) != 1:
            continue
        sum_sub = sum_exp_call.args[0]
        if not (isinstance(sum_sub, ast.BinOp) and isinstance(sum_sub.op, ast.Sub)):
            continue
        if _name(sum_sub.left) != values_name:
            continue
        if not isinstance(sum_sub.right, ast.Subscript):
            continue
        if _name(sum_sub.right.value) != mi_next_name:
            continue
        if _expr_unparse(sum_sub.right.slice) not in {
            "(slice(None, None, None), None)",
            "(:, None)",
        }:
            continue

        # mi = mi_next
        if _single_target_name(b4) != mi_name:
            continue
        assert isinstance(b4, ast.Assign)
        if _name(b4.value) != mi_next_name:
            continue

        # Match the consume loop body too — at minimum it must reference
        # ``mi`` and ``di`` (so the names stay live across the splice) and
        # use the same ``tile_n`` index.
        body2 = _strip_doc(list(loop2.body))
        consume_text = ast.unparse(ast.Module(body=body2, type_ignores=[]))  # type: ignore[arg-type]
        if mi_name not in consume_text or di_name not in consume_text:
            continue
        if tile_n_name not in consume_text:
            continue

        info: dict[str, object] = {
            "tile_m_name": tile_m_name,
            "tile_n_name": tile_n_name,
            "mi_name": mi_name,
            "di_name": di_name,
            "mi_next_name": mi_next_name,
            "local_amax_name": local_amax_name,
            "values_name": values_name,
            "src_name": src_name,
            "loop1": loop1,
            "loop2": loop2,
            "s_mi": s_mi,
            "s_di": s_di,
            "di_init_value_repr": di_init_value_repr,
        }
        return i, info
    return None


def _build_max_loop(
    template_loop: ast.For,
    *,
    tile_m_name: str,
    tile_n_name: str,
    mi_name: str,
    local_amax_name: str,
    values_name: str,
    src_name: str,
) -> ast.For:
    """Build the max-only pass:

    for tile_n in hl.tile(...):
        values = x[tile_m, tile_n]
        local_amax = torch.amax(values, dim=1)
        mi = torch.maximum(mi, local_amax)
    """
    values_assign = create(
        ast.Assign,
        targets=[create(ast.Name, id=values_name, ctx=ast.Store())],
        value=create(
            ast.Subscript,
            value=create(ast.Name, id=src_name, ctx=ast.Load()),
            slice=create(
                ast.Tuple,
                elts=[
                    create(ast.Name, id=tile_m_name, ctx=ast.Load()),
                    create(ast.Name, id=tile_n_name, ctx=ast.Load()),
                ],
                ctx=ast.Load(),
            ),
            ctx=ast.Load(),
        ),
        type_comment=None,
    )
    local_amax_assign = create(
        ast.Assign,
        targets=[create(ast.Name, id=local_amax_name, ctx=ast.Store())],
        value=create(
            ast.Call,
            func=create(
                ast.Attribute,
                value=create(ast.Name, id="torch", ctx=ast.Load()),
                attr="amax",
                ctx=ast.Load(),
            ),
            args=[create(ast.Name, id=values_name, ctx=ast.Load())],
            keywords=[
                create(
                    ast.keyword,
                    arg="dim",
                    value=create(ast.Constant, value=1, kind=None),
                )
            ],
        ),
        type_comment=None,
    )
    mi_update = create(
        ast.Assign,
        targets=[create(ast.Name, id=mi_name, ctx=ast.Store())],
        value=create(
            ast.Call,
            func=create(
                ast.Attribute,
                value=create(ast.Name, id="torch", ctx=ast.Load()),
                attr="maximum",
                ctx=ast.Load(),
            ),
            args=[
                create(ast.Name, id=mi_name, ctx=ast.Load()),
                create(ast.Name, id=local_amax_name, ctx=ast.Load()),
            ],
            keywords=[],
        ),
        type_comment=None,
    )
    return _make_for_loop(template_loop, [values_assign, local_amax_assign, mi_update])


def _build_sum_loop(
    template_loop: ast.For,
    *,
    tile_m_name: str,
    tile_n_name: str,
    mi_name: str,
    di_name: str,
    values_name: str,
    src_name: str,
) -> ast.For:
    """Build the sum-only pass:

    for tile_n in hl.tile(...):
        values = x[tile_m, tile_n]
        di = di + torch.exp(values - mi[:, None]).sum(dim=1)
    """
    values_assign = create(
        ast.Assign,
        targets=[create(ast.Name, id=values_name, ctx=ast.Store())],
        value=create(
            ast.Subscript,
            value=create(ast.Name, id=src_name, ctx=ast.Load()),
            slice=create(
                ast.Tuple,
                elts=[
                    create(ast.Name, id=tile_m_name, ctx=ast.Load()),
                    create(ast.Name, id=tile_n_name, ctx=ast.Load()),
                ],
                ctx=ast.Load(),
            ),
            ctx=ast.Load(),
        ),
        type_comment=None,
    )
    # mi[:, None]
    mi_broadcast = create(
        ast.Subscript,
        value=create(ast.Name, id=mi_name, ctx=ast.Load()),
        slice=create(
            ast.Tuple,
            elts=[
                create(
                    ast.Slice,
                    lower=None,
                    upper=None,
                    step=None,
                ),
                create(ast.Constant, value=None, kind=None),
            ],
            ctx=ast.Load(),
        ),
        ctx=ast.Load(),
    )
    # values - mi[:, None]
    sub_expr = create(
        ast.BinOp,
        left=create(ast.Name, id=values_name, ctx=ast.Load()),
        op=create(ast.Sub),
        right=mi_broadcast,
    )
    # torch.exp(values - mi[:, None])
    exp_call = create(
        ast.Call,
        func=create(
            ast.Attribute,
            value=create(ast.Name, id="torch", ctx=ast.Load()),
            attr="exp",
            ctx=ast.Load(),
        ),
        args=[sub_expr],
        keywords=[],
    )
    # torch.exp(...).sum(dim=1)
    sum_call = create(
        ast.Call,
        func=create(
            ast.Attribute,
            value=exp_call,
            attr="sum",
            ctx=ast.Load(),
        ),
        args=[],
        keywords=[
            create(
                ast.keyword,
                arg="dim",
                value=create(ast.Constant, value=1, kind=None),
            )
        ],
    )
    # di = di + torch.exp(...).sum(dim=1)
    di_update = create(
        ast.Assign,
        targets=[create(ast.Name, id=di_name, ctx=ast.Store())],
        value=create(
            ast.BinOp,
            left=create(ast.Name, id=di_name, ctx=ast.Load()),
            op=create(ast.Add),
            right=sum_call,
        ),
        type_comment=None,
    )
    return _make_for_loop(template_loop, [values_assign, di_update])


def _rewrite_outer_body(outer_body: list[ast.stmt]) -> tuple[list[ast.stmt], bool]:
    """If ``outer_body`` matches the online softmax pattern, return the
    rewritten body and ``True``.  Otherwise return ``(outer_body, False)``.
    """
    match = _detect_online_softmax(outer_body)
    if match is None:
        return outer_body, False
    start, info = match
    # Build the 3-pass replacement.
    tile_m_name = info["tile_m_name"]
    tile_n_name = info["tile_n_name"]
    mi_name = info["mi_name"]
    di_name = info["di_name"]
    local_amax_name = info["local_amax_name"]
    values_name = info["values_name"]
    src_name = info["src_name"]
    loop1 = info["loop1"]
    loop2 = info["loop2"]
    s_mi = info["s_mi"]
    s_di = info["s_di"]
    assert isinstance(tile_m_name, str)
    assert isinstance(tile_n_name, str)
    assert isinstance(mi_name, str)
    assert isinstance(di_name, str)
    assert isinstance(local_amax_name, str)
    assert isinstance(values_name, str)
    assert isinstance(src_name, str)
    assert isinstance(loop1, ast.For)
    assert isinstance(loop2, ast.For)
    assert isinstance(s_mi, ast.Assign)
    assert isinstance(s_di, ast.Assign)

    max_loop = _build_max_loop(
        loop1,
        tile_m_name=tile_m_name,
        tile_n_name=tile_n_name,
        mi_name=mi_name,
        local_amax_name=local_amax_name,
        values_name=values_name,
        src_name=src_name,
    )
    sum_loop = _build_sum_loop(
        loop1,
        tile_m_name=tile_m_name,
        tile_n_name=tile_n_name,
        mi_name=mi_name,
        di_name=di_name,
        values_name=values_name,
        src_name=src_name,
    )

    # Splice: keep mi init, move di init AFTER the max loop, drop the
    # online-merge loop, insert max-only then sum-only loops, keep the
    # consume loop (loop2) as-is.
    new_body = list(outer_body)
    # Replace the 4-stmt window [s_mi, s_di, loop1, loop2] with
    # [s_mi, max_loop, s_di (deepcopied), sum_loop, loop2].
    s_di_copy = _ext_copy(s_di)
    # pyrefly: ignore [bad-assignment]
    replacement: list[ast.stmt] = [s_mi, max_loop, s_di_copy, sum_loop, loop2]
    new_body[start : start + 4] = replacement
    return new_body, True


class _OnlineToThreePassTransformer(ast.NodeTransformer):
    """Walks the host function body looking for ``for tile_m in hl.tile(...)``
    loops whose bodies match the online softmax pattern; rewrites those
    bodies in-place to the 3-pass form.
    """

    def __init__(self) -> None:
        super().__init__()
        self.rewrites = 0

    def visit_For(self, node: ast.For) -> ast.AST:
        # Recurse first so nested patterns get rewritten.
        self.generic_visit(node)
        # Only the outermost device for-loop (LoopType.GRID) carries the
        # online softmax pattern in its body.  Walk both GRID and DEVICE
        # loops to also catch nested cases — match-or-pass-through is
        # safe because detection is conservative.
        new_body, fired = _rewrite_outer_body(list(node.body))
        if fired:
            self.rewrites += 1
            node.body = new_body
        return node


def _min_n_for_rewrite() -> int:
    """The reduction-axis extent (``N``) at or above which the 3-pass form
    is profitable.  Defaults to 2048; set ``HELION_ONLINE_TO_3PASS_MIN_N``
    to override (e.g. ``0`` to always fire, useful for A/B tests).
    """
    val = os.environ.get("HELION_ONLINE_TO_3PASS_MIN_N")
    if val is None:
        return 2048
    try:
        return int(val)
    except ValueError:
        return 2048


def _reduction_axis_extent(func: HostFunction) -> int | None:
    """Return a conservative estimate of the kernel's reduction-axis extent.

    For ``softmax_two_pass(x)`` with ``x.shape = (m, n)`` the reduction
    axis is ``n`` (the innermost dim, since the outer ``for tile_m`` is
    over rows and the inner ``for tile_n`` is the reduction sweep).
    We approximate the reduction-axis extent as the LAST dim of the
    LARGEST tensor input (by total element count).  This works for the
    canonical 2D softmax shape and is conservative for kernels that
    don't follow this convention (their detected pattern still won't
    rewrite if the last-dim extent is below the cutoff).
    """
    import torch  # local import — keep module-level imports minimal

    chosen: tuple[int, int] | None = None  # (num_elements, last_dim)
    for arg in func.params.arguments.values():
        if not isinstance(arg, torch.Tensor):
            continue
        sizes = arg.size()
        if not sizes:
            continue
        # Only consider tensors whose dims are all static ints.
        if not all(isinstance(d, int) for d in sizes):
            continue
        last = int(sizes[-1])
        total = 1
        for d in sizes:
            total *= int(d)
        if chosen is None or total > chosen[0]:
            chosen = (total, last)
    return None if chosen is None else chosen[1]


def rewrite_online_to_3pass(func: HostFunction) -> bool:
    """Rewrite online softmax patterns in ``func.body`` to the 3-pass form.

    Returns True if any rewrite fired.  Gated externally by:

    * ``HELION_DISABLE_ONLINE_TO_3PASS=1`` — skip the pass entirely.
    * ``HELION_ONLINE_TO_3PASS_MIN_N`` — minimum reduction-axis extent
      for the rewrite to apply (default 2048).
    """
    if os.environ.get("HELION_DISABLE_ONLINE_TO_3PASS") == "1":
        return False
    min_n = _min_n_for_rewrite()
    if min_n > 0:
        extent = _reduction_axis_extent(func)
        if extent is not None and extent < min_n:
            return False
    transformer = _OnlineToThreePassTransformer()
    new_body: list[ast.stmt] = []
    for stmt in func.body:
        result = transformer.visit(stmt)
        if isinstance(result, list):
            new_body.extend(result)
        else:
            assert isinstance(result, ast.stmt)
            new_body.append(result)
    func.body = new_body
    return transformer.rewrites > 0
