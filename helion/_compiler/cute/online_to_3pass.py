"""Rewrite a closed online-softmax pattern into three CuTe sweeps.

The recognized initializers use FP32. An explicit FP32 input conversion stays
before every reduction and subtraction, and the original output conversion
stays in the consume loop. Finite sums are reassociated: the pass preserves
the existing softmax tolerance contract, not bitwise equality. It does not
enable fast_math or change its default. In particular, leading all--inf
logical tiles must still poison the denominator, and an empty reduction
must leave it at positive zero.

The rewrite requires pure signatures and iterators, a canonical consume
loop, and no observations of its state or temporaries outside the matched
region. ``HELION_DISABLE_ONLINE_TO_3PASS=1`` disables it; the optional
``HELION_ONLINE_TO_3PASS_MIN_N`` threshold defaults to zero.


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

For finite inputs, the three sweeps compute the same real-arithmetic result:
  * max-pass final mi = max over all tiles of local_amax — equivalent to
    the running maximum once the loop has visited every tile.
  * sum-pass final di = sum over all tiles of sum(exp(values - mi_final))
    — equivalent to the running rescaled sum once mi has reached its
    final value.

The max sweep also reduces the first logical tile independently. Its result
preserves the online denominator's NaN when that first tile is all -inf,
without placing a consumer between a partial lane reduction and its complete
logical-tile reduction. Later nonfinite inputs already poison the exp sum.
This pass runs on source AST before tracing, only for the CuTe backend.
"""

from __future__ import annotations

import ast
import builtins
import inspect
import math
import os
from typing import TYPE_CHECKING

import torch

from ..ast_extension import ExtendedAST
from ..ast_extension import create
import helion.language as hl

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
    """Accept a zero-based extent and an optional pure, invariant block size."""
    if not _is_hl_call(node, "tile"):
        return False
    assert isinstance(node, ast.Call)
    if len(node.args) != 1 or not _integer_atom(node.args[0], minimum=0):
        return False
    if not node.keywords:
        return True
    return (
        len(node.keywords) == 1
        and node.keywords[0].arg == "block_size"
        and _integer_atom(node.keywords[0].value, minimum=1)
    )


def _integer_atom(node: ast.AST, *, minimum: int) -> bool:
    # hl.tile validates the type of a named extent/block size during tracing.
    # No arithmetic, calls, subscriptions, or attribute evaluation is duplicated.
    return isinstance(node, ast.Name) or (
        isinstance(node, ast.Constant)
        and type(node.value) is int
        and node.value >= minimum
    )


def _fp32_dtype(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and _name(node.value) == "torch"
        and node.attr == "float32"
    )


def _initializer_tile(node: ast.expr, *, maximum: bool) -> str | None:
    """Match the complete FP32 initializer, including its positive-zero sign."""
    if not isinstance(node, ast.Call):
        return None
    is_full = _is_hl_call(node, "full")
    if not is_full and (maximum or not _is_hl_call(node, "zeros")):
        return None
    if len(node.args) != (2 if is_full else 1):
        return None
    if not (
        len(node.keywords) == 1
        and node.keywords[0].arg == "dtype"
        and _fp32_dtype(node.keywords[0].value)
    ):
        return None
    shape = node.args[0]
    if not isinstance(shape, ast.List) or len(shape.elts) != 1:
        return None
    tile = _name(shape.elts[0])
    if tile is None:
        return None
    if is_full:
        fill = node.args[1]
        if maximum:
            if not (
                isinstance(fill, ast.Call)
                and _name(fill.func) == "float"
                and len(fill.args) == 1
                and not fill.keywords
                and isinstance(fill.args[0], ast.Constant)
                and fill.args[0].value == "-inf"
            ) and not (
                isinstance(fill, ast.UnaryOp)
                and isinstance(fill.op, ast.USub)
                and isinstance(fill.operand, ast.Attribute)
                and _name(fill.operand.value) == "math"
                and fill.operand.attr == "inf"
            ):
                return None
        elif not (
            isinstance(fill, ast.Constant)
            and type(fill.value) in (int, float)
            and fill.value == 0
            and math.copysign(1.0, fill.value) > 0
        ):
            return None
    return tile


def _typed_load(node: ast.expr) -> ast.Subscript | None:
    """Accept a direct load or its explicit FP32 conversion."""
    if isinstance(node, ast.Subscript):
        return node
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "to"
        and isinstance(node.func.value, ast.Subscript)
    ):
        return None
    if len(node.args) == 1 and not node.keywords:
        dtype = node.args[0]
    elif not node.args and len(node.keywords) == 1 and node.keywords[0].arg == "dtype":
        dtype = node.keywords[0].value
    else:
        return None
    if not _fp32_dtype(dtype):
        return None
    return node.func.value


def _sum_argument(node: ast.expr) -> ast.expr | None:
    """Match only a row sum with its default dtype and no output argument."""
    return _row_reduction_argument(node, "sum", allow_method=True)


def _row_reduction_argument(
    node: ast.expr, operation: str, *, allow_method: bool = False
) -> ast.expr | None:
    if not isinstance(node, ast.Call):
        return None
    if _is_torch_call(node, operation):
        if not node.args:
            return None
        value = node.args[0]
        dims = node.args[1:]
    elif (
        allow_method
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == operation
    ):
        value = node.func.value
        dims = node.args
    else:
        return None
    if len(dims) == 1 and not node.keywords:
        dim = dims[0]
    elif not dims and len(node.keywords) == 1 and node.keywords[0].arg == "dim":
        dim = node.keywords[0].value
    else:
        return None
    return (
        value
        if isinstance(dim, ast.Constant) and type(dim.value) is int and dim.value == 1
        else None
    )


def _broadcast(node: ast.AST, name: str) -> bool:
    if not isinstance(node, ast.Subscript) or _name(node.value) != name:
        return False
    index = node.slice
    if not isinstance(index, ast.Tuple) or len(index.elts) != 2:
        return False
    rows, new_axis = index.elts
    if not isinstance(new_axis, ast.Constant) or new_axis.value is not None:
        return False
    if isinstance(rows, ast.Slice):
        return rows.lower is None and rows.upper is None and rows.step is None
    # Host preprocessing spells ':' as slice(None, None, None).
    return (
        isinstance(rows, ast.Call)
        and _name(rows.func) == "slice"
        and len(rows.args) == 3
        and not rows.keywords
        and all(
            isinstance(arg, ast.Constant) and arg.value is None for arg in rows.args
        )
    )


def _tile_subscript(node: ast.AST, tile_m: str, tile_n: str) -> str | None:
    if not isinstance(node, ast.Subscript):
        return None
    index = node.slice
    if not (
        isinstance(index, ast.Tuple)
        and len(index.elts) == 2
        and _name(index.elts[0]) == tile_m
        and _name(index.elts[1]) == tile_n
    ):
        return None
    return _name(node.value)


def _consume_output(
    loop: ast.For,
    *,
    values_assign: ast.Assign,
    tile_m: str,
    tile_n: str,
    mi: str,
    di: str,
) -> str | None:
    """Require the unchanged load / pure normalization / owned-tile store."""
    body = _strip_doc(list(loop.body))
    if len(body) != 2 or not isinstance(body[0], ast.Assign):
        return None
    load, store = body
    assert isinstance(load, ast.Assign)
    if (
        _single_target_name(load) != _single_target_name(values_assign)
        or _expr_unparse(load.value) != _expr_unparse(values_assign.value)
        or not isinstance(store, ast.Assign)
        or len(store.targets) != 1
    ):
        return None
    output = _tile_subscript(store.targets[0], tile_m, tile_n)
    if output is None:
        return None
    normalized = store.value
    if isinstance(normalized, ast.Call):
        if not (
            isinstance(normalized.func, ast.Attribute) and normalized.func.attr == "to"
        ):
            return None
        if len(normalized.args) == 1 and not normalized.keywords:
            dtype = normalized.args[0]
        elif (
            not normalized.args
            and len(normalized.keywords) == 1
            and normalized.keywords[0].arg == "dtype"
        ):
            dtype = normalized.keywords[0].value
        else:
            return None
        if not (
            isinstance(dtype, ast.Attribute)
            and _name(dtype.value) == output
            and dtype.attr == "dtype"
        ):
            return None
        normalized = normalized.func.value
    if not (
        isinstance(normalized, ast.BinOp)
        and isinstance(normalized.op, ast.Div)
        and _broadcast(normalized.right, di)
        and _is_torch_call(normalized.left, "exp")
    ):
        return None
    numerator = normalized.left
    assert isinstance(numerator, ast.Call)
    if len(numerator.args) != 1 or numerator.keywords:
        return None
    shifted = numerator.args[0]
    if not (
        isinstance(shifted, ast.BinOp)
        and isinstance(shifted.op, ast.Sub)
        and _name(shifted.left) == _single_target_name(values_assign)
        and _broadcast(shifted.right, mi)
    ):
        return None
    return output


def _direct_names(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.arg):
        return (node.arg,)
    if isinstance(node, (ast.Global, ast.Nonlocal)):
        return tuple(node.names)
    if isinstance(node, ast.alias):
        return (node.asname or node.name.split(".", 1)[0],)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return (node.name,)
    if isinstance(node, ast.ExceptHandler) and node.name is not None:
        return (node.name,)
    if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name is not None:
        return (node.name,)
    if isinstance(node, ast.MatchMapping) and node.rest is not None:
        return (node.rest,)
    return ()


def _names(node: ast.AST) -> set[str]:
    """Include bindings as well as reads, even bindings without an ast.Name."""
    return {name for child in ast.walk(node) for name in _direct_names(child)}


def _outside_names(body: list[ast.stmt], region: list[ast.stmt]) -> set[str]:
    """Names outside the region, including a later outer iteration's prefix."""
    excluded = {id(stmt) for stmt in region}

    class OutsideNames(ast.NodeVisitor):
        def __init__(self) -> None:
            self.names: set[str] = set()

        def visit(self, node: ast.AST) -> None:
            if id(node) in excluded:
                return
            self.names.update(_direct_names(node))
            super().visit(node)

    visitor = OutsideNames()
    for stmt in body:
        visitor.visit(stmt)
    return visitor.names


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
    *,
    scope_body: list[ast.stmt] | None = None,
    parameter_names: frozenset[str] = frozenset(),
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
    if scope_body is None:
        scope_body = outer_body
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

        tile_m_name = _initializer_tile(s_mi.value, maximum=True)
        if (
            tile_m_name is None
            or _initializer_tile(s_di.value, maximum=False) != tile_m_name
        ):
            continue

        # First inner for-loop: online merge
        loop1 = _ensure_for_extended(s_loop1)
        loop2 = _ensure_for_extended(s_loop2)
        if loop1 is None or loop2 is None:
            continue
        if not _is_tile_call(loop1.iter) or not _is_tile_call(loop2.iter):
            continue
        assert isinstance(loop1.iter, ast.Call)
        assert isinstance(loop2.iter, ast.Call)
        if loop1.orelse or loop2.orelse:
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
        load = _typed_load(b0.value)
        if load is None:
            continue
        src_name = _tile_subscript(load, tile_m_name, tile_n_name)
        if src_name is None:
            continue

        # local_amax = torch.amax(values, dim=1)
        local_amax_name = _single_target_name(b1)
        if local_amax_name is None:
            continue
        assert isinstance(b1, ast.Assign)
        if _name(_row_reduction_argument(b1.value, "amax")) != values_name:
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
        if len(mx_call.args) != 2 or mx_call.keywords:
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
        if len(exp_call.args) != 1 or exp_call.keywords:
            continue
        sub_expr = exp_call.args[0]
        if not (isinstance(sub_expr, ast.BinOp) and isinstance(sub_expr.op, ast.Sub)):
            continue
        if _name(sub_expr.left) != mi_name or _name(sub_expr.right) != mi_next_name:
            continue
        # right: torch.exp(values - mi_next[:, None]).sum(dim=1)
        sum_target = _sum_argument(right)
        if sum_target is None:
            continue
        if not _is_torch_call(sum_target, "exp"):
            continue
        sum_exp_call = sum_target
        assert isinstance(sum_exp_call, ast.Call)
        if len(sum_exp_call.args) != 1 or sum_exp_call.keywords:
            continue
        sum_sub = sum_exp_call.args[0]
        if not (isinstance(sum_sub, ast.BinOp) and isinstance(sum_sub.op, ast.Sub)):
            continue
        if _name(sum_sub.left) != values_name:
            continue
        if not _broadcast(sum_sub.right, mi_next_name):
            continue

        # mi = mi_next
        if _single_target_name(b4) != mi_name:
            continue
        assert isinstance(b4, ast.Assign)
        if _name(b4.value) != mi_next_name:
            continue

        output_name = _consume_output(
            loop2,
            values_assign=b0,
            tile_m=tile_m_name,
            tile_n=tile_n_name,
            mi=mi_name,
            di=di_name,
        )
        if output_name is None:
            continue
        local_names = {mi_name, di_name, values_name, local_amax_name, mi_next_name}
        if len(local_names) != 5:
            continue
        if len({tile_m_name, tile_n_name, src_name}) != 3:
            continue
        if output_name in {tile_m_name, tile_n_name}:
            continue
        iterator_names = _names(loop1.iter) - {"hl"}
        if iterator_names & {tile_m_name, tile_n_name, src_name, output_name}:
            continue
        invariant_names = iterator_names | {
            tile_m_name,
            tile_n_name,
            src_name,
            output_name,
            "hl",
            "torch",
            "float",
            "slice",
            "math",
        }
        if local_names & invariant_names:
            continue
        # This also rejects prefix reads that observe the preceding row's
        # locals, and uses after the outer loop. Neither is visible to a
        # suffix-only liveness check. Empty loops retain their old bindings.
        region_names = local_names | {tile_n_name}
        if region_names & (
            parameter_names | _outside_names(scope_body, outer_body[i : i + 4])
        ):
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
            "values_expr": b0.value,
            "loop1": loop1,
            "loop2": loop2,
            "s_mi": s_mi,
            "s_di": s_di,
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
    values_expr: ast.expr,
    first_max_name: str,
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
        value=_ext_copy(values_expr),
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
    first_local_name = first_max_name + "_local"
    first_local_assign = create(
        ast.Assign,
        targets=[create(ast.Name, id=first_local_name, ctx=ast.Store())],
        value=create(
            ast.Call,
            func=create(
                ast.Attribute,
                value=create(ast.Name, id="torch", ctx=ast.Load()),
                attr="amax",
                ctx=ast.Load(),
            ),
            args=[
                create(
                    ast.Call,
                    func=create(
                        ast.Attribute,
                        value=create(ast.Name, id="torch", ctx=ast.Load()),
                        attr="where",
                        ctx=ast.Load(),
                    ),
                    args=[
                        create(
                            ast.Call,
                            func=create(
                                ast.Attribute,
                                value=create(ast.Name, id="hl", ctx=ast.Load()),
                                attr="full",
                                ctx=ast.Load(),
                            ),
                            args=[
                                create(
                                    ast.List,
                                    elts=[
                                        create(
                                            ast.Name, id=tile_m_name, ctx=ast.Load()
                                        ),
                                        create(
                                            ast.Name, id=tile_n_name, ctx=ast.Load()
                                        ),
                                    ],
                                    ctx=ast.Load(),
                                ),
                                create(
                                    ast.Compare,
                                    left=create(
                                        ast.Attribute,
                                        value=create(
                                            ast.Name, id=tile_n_name, ctx=ast.Load()
                                        ),
                                        attr="begin",
                                        ctx=ast.Load(),
                                    ),
                                    ops=[create(ast.Eq)],
                                    comparators=[
                                        create(ast.Constant, value=0, kind=None)
                                    ],
                                ),
                            ],
                            keywords=[
                                create(
                                    ast.keyword,
                                    arg="dtype",
                                    value=create(
                                        ast.Attribute,
                                        value=create(
                                            ast.Name, id="torch", ctx=ast.Load()
                                        ),
                                        attr="bool",
                                        ctx=ast.Load(),
                                    ),
                                )
                            ],
                        ),
                        create(ast.Name, id=values_name, ctx=ast.Load()),
                        create(ast.Constant, value=float("-inf"), kind=None),
                    ],
                    keywords=[],
                ),
            ],
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
    first_max_update = create(
        ast.Assign,
        targets=[create(ast.Name, id=first_max_name, ctx=ast.Store())],
        value=create(
            ast.Call,
            func=create(
                ast.Attribute,
                value=create(ast.Name, id="torch", ctx=ast.Load()),
                attr="maximum",
                ctx=ast.Load(),
            ),
            args=[
                create(ast.Name, id=first_max_name, ctx=ast.Load()),
                create(ast.Name, id=first_local_name, ctx=ast.Load()),
            ],
            keywords=[],
        ),
        type_comment=None,
    )
    return _make_for_loop(
        template_loop,
        [
            values_assign,
            local_amax_assign,
            mi_update,
            first_local_assign,
            first_max_update,
        ],
    )


def _build_sum_loop(
    template_loop: ast.For,
    *,
    mi_name: str,
    di_name: str,
    values_name: str,
    values_expr: ast.expr,
) -> ast.For:
    """Build the sum-only pass:

    for tile_n in hl.tile(...):
        values = x[tile_m, tile_n]
        di = di + torch.exp(values - mi[:, None]).sum(dim=1)
    """
    values_assign = create(
        ast.Assign,
        targets=[create(ast.Name, id=values_name, ctx=ast.Store())],
        value=_ext_copy(values_expr),
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


def _rewrite_outer_body(
    outer_body: list[ast.stmt],
    *,
    scope_body: list[ast.stmt] | None = None,
    parameter_names: frozenset[str] = frozenset(),
    used_names: set[str] | None = None,
) -> tuple[list[ast.stmt], bool]:
    """If ``outer_body`` matches the online softmax pattern, return the
    rewritten body and ``True``.  Otherwise return ``(outer_body, False)``.
    """
    match = _detect_online_softmax(
        outer_body, scope_body=scope_body, parameter_names=parameter_names
    )
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
    values_expr = info["values_expr"]
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
    assert isinstance(values_expr, ast.expr)
    assert isinstance(loop1, ast.For)
    assert isinstance(loop1.iter, ast.Call)
    assert isinstance(loop2, ast.For)
    assert isinstance(s_mi, ast.Assign)
    assert isinstance(s_di, ast.Assign)

    if used_names is None:
        used_names = set(parameter_names)
        for stmt in scope_body if scope_body is not None else outer_body:
            used_names.update(_names(stmt))
    first_max_name = "_helion_first_tile_max"
    while first_max_name in used_names or first_max_name + "_local" in used_names:
        first_max_name += "_"
    used_names.update((first_max_name, first_max_name + "_local"))
    first_max_init = _ext_copy(s_mi)
    assert isinstance(first_max_init, ast.Assign)
    first_max_init.targets = [create(ast.Name, id=first_max_name, ctx=ast.Store())]

    max_loop = _build_max_loop(
        loop1,
        tile_m_name=tile_m_name,
        tile_n_name=tile_n_name,
        mi_name=mi_name,
        local_amax_name=local_amax_name,
        values_name=values_name,
        values_expr=values_expr,
        first_max_name=first_max_name,
    )
    sum_loop = _build_sum_loop(
        loop1,
        mi_name=mi_name,
        di_name=di_name,
        values_name=values_name,
        values_expr=values_expr,
    )

    # Splice: keep mi init, move di init AFTER the max loop, drop the
    # online-merge loop, insert max-only then sum-only loops, keep the
    # consume loop (loop2) as-is.
    new_body = list(outer_body)
    # Replace the 4-stmt window [s_mi, s_di, loop1, loop2] with
    # [s_mi, max_loop, s_di (deepcopied), sum_loop, loop2].
    s_di_copy = _ext_copy(s_di)
    poison_update = create(
        ast.Assign,
        targets=[create(ast.Name, id=di_name, ctx=ast.Store())],
        value=create(
            ast.Call,
            func=create(
                ast.Attribute,
                value=create(ast.Name, id="torch", ctx=ast.Load()),
                attr="where",
                ctx=ast.Load(),
            ),
            args=[
                create(
                    ast.BinOp,
                    left=create(
                        ast.Compare,
                        left=create(ast.Name, id=first_max_name, ctx=ast.Load()),
                        ops=[create(ast.Eq)],
                        comparators=[
                            create(ast.Constant, value=float("-inf"), kind=None)
                        ],
                    ),
                    op=create(ast.BitAnd),
                    right=create(
                        ast.Call,
                        func=create(
                            ast.Attribute,
                            value=create(ast.Name, id="hl", ctx=ast.Load()),
                            attr="full",
                            ctx=ast.Load(),
                        ),
                        args=[
                            create(
                                ast.List,
                                elts=[create(ast.Name, id=tile_m_name, ctx=ast.Load())],
                                ctx=ast.Load(),
                            ),
                            create(
                                ast.Compare,
                                left=_ext_copy(loop1.iter.args[0]),
                                ops=[create(ast.Gt)],
                                comparators=[create(ast.Constant, value=0, kind=None)],
                            ),
                        ],
                        keywords=[
                            create(
                                ast.keyword,
                                arg="dtype",
                                value=create(
                                    ast.Attribute,
                                    value=create(ast.Name, id="torch", ctx=ast.Load()),
                                    attr="bool",
                                    ctx=ast.Load(),
                                ),
                            )
                        ],
                    ),
                ),
                create(ast.Constant, value=float("nan"), kind=None),
                create(ast.Name, id=di_name, ctx=ast.Load()),
            ],
            keywords=[],
        ),
        type_comment=None,
    )
    # Apply the poison after summation. NaN absorbs every FP32 addition, so
    # selecting it after the pure sum has the same observable result as
    # initializing the sum with it. A false predicate keeps the original
    # positive-zero initializer and sum order, including the empty-axis case.
    # This also leaves the independent sum in its ordinary reduction form.
    # pyrefly: ignore [bad-assignment]
    replacement: list[ast.stmt] = [
        s_mi,
        first_max_init,
        max_loop,
        s_di_copy,
        sum_loop,
        poison_update,
        loop2,
    ]
    new_body[start : start + 4] = replacement
    return new_body, True


class _OnlineToThreePassTransformer(ast.NodeTransformer):
    """Walks the host function body looking for ``for tile_m in hl.tile(...)``
    loops whose bodies match the online softmax pattern; rewrites those
    bodies in-place to the 3-pass form.
    """

    def __init__(
        self, scope_body: list[ast.stmt], parameter_names: frozenset[str]
    ) -> None:
        super().__init__()
        self.rewrites = 0
        self.scope_body = scope_body
        self.parameter_names = parameter_names
        self.used_names = set(parameter_names)
        for stmt in scope_body:
            self.used_names.update(_names(stmt))

    def visit_For(self, node: ast.For) -> ast.AST:
        # Recurse first so nested patterns get rewritten.
        self.generic_visit(node)
        # Only the outermost device for-loop (LoopType.GRID) carries the
        # online softmax pattern in its body.  Walk both GRID and DEVICE
        # loops to also catch nested cases — match-or-pass-through is
        # safe because detection is conservative.
        new_body, fired = _rewrite_outer_body(
            list(node.body),
            scope_body=self.scope_body,
            parameter_names=self.parameter_names,
            used_names=self.used_names,
        )
        if fired:
            self.rewrites += 1
            node.body = new_body
        return node


def _min_n_for_rewrite() -> int:
    """Optional reduction-axis threshold; the existing default is zero."""
    val = os.environ.get("HELION_ONLINE_TO_3PASS_MIN_N")
    if val is None:
        return 0
    try:
        return int(val)
    except ValueError:
        return 0


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
      for the rewrite to apply (default zero).
    """
    if os.environ.get("HELION_DISABLE_ONLINE_TO_3PASS") == "1":
        return False
    # Pure-looking calls must actually resolve to these modules/builtins.
    # A parameter or local binding also shadows the corresponding global.
    closure = inspect.getclosurevars(func.fn)
    bindings = {**closure.builtins, **closure.globals, **closure.nonlocals}
    if bindings.get("torch") is not torch or bindings.get("hl") is not hl:
        return False
    for name, expected in (
        ("float", builtins.float),
        ("slice", builtins.slice),
        ("math", math),
    ):
        if bindings.get(name, expected) is not expected:
            return False
    reserved = {"torch", "hl", "float", "slice", "math"}
    parameter_names = frozenset(
        node.arg for node in ast.walk(func.args) if isinstance(node, ast.arg)
    )
    if parameter_names & reserved or any(
        isinstance(node, ast.Name)
        and not isinstance(node.ctx, ast.Load)
        and node.id in reserved
        for stmt in func.body
        for node in ast.walk(stmt)
    ):
        return False
    min_n = _min_n_for_rewrite()
    if min_n > 0:
        extent = _reduction_axis_extent(func)
        if extent is not None and extent < min_n:
            return False
    transformer = _OnlineToThreePassTransformer(func.body, parameter_names)
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
