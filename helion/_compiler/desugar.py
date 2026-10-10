"""Generic frontend desugaring pass.

Rewrites user-facing sugar into the core forms the rest of the compiler
understands.  Runs once between parse and static loop unrolling; each sugar
is a high-level condition branch in ``_desugar_statement``, so adding new
sugar only ever touches this file.

Current sugar:
    ``with hl.device_scope(): BODY`` becomes
    ``for _ in hl.grid(1): BODY``
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from .. import exc
from ..language.device_scope import device_scope
from .ast_extension import ExtendedAST
from .ast_extension import create

if TYPE_CHECKING:
    from .host_function import HostFunction


def _resolve_expr(func: HostFunction, node: ast.expr) -> object | None:
    """Resolve a Name/Attribute chain through the kernel's module globals."""
    if isinstance(node, ast.Name):
        return func.fn.__globals__.get(node.id)
    if isinstance(node, ast.Attribute):
        base = _resolve_expr(func, node.value)
        if base is None:
            return None
        return getattr(base, node.attr, None)
    return None


def desugar(func: HostFunction) -> None:
    """Rewrite frontend sugar (e.g. `with hl.device_scope():`) to core forms."""
    func.body = [_desugar_statement(func, stmt) for stmt in func.body]


def _desugar_statement(func: HostFunction, stmt: ast.stmt) -> ast.stmt:
    if isinstance(stmt, ast.With) and (
        (item := _device_scope_item(func, stmt)) is not None
    ):
        return _rewrite_device_scope(stmt, item)
    return stmt


def _device_scope_item(func: HostFunction, node: ast.With) -> ast.withitem | None:
    """Return the with-item whose context is an hl.device_scope() call, if any."""
    for item in node.items:
        ctx = item.context_expr
        if (
            isinstance(ctx, ast.Call)
            and isinstance(ctx.func, (ast.Name, ast.Attribute))
            and _resolve_expr(func, ctx.func) is device_scope
        ):
            if not isinstance(ctx.func, ast.Attribute):
                raise exc.DeviceScopeInvalidUsage(
                    "use hl.device_scope() with the language module imported "
                    "(e.g. `import helion.language as hl`)"
                )
            return item
    return None


def _rewrite_device_scope(node: ast.With, item: ast.withitem) -> ast.For:
    """Rewrite `with hl.device_scope(): BODY` into `for _ in hl.grid(1): BODY`."""
    if len(node.items) != 1:
        raise exc.DeviceScopeInvalidUsage(
            "it must be the only context manager in the with statement"
        )
    if item.optional_vars is not None:
        raise exc.DeviceScopeInvalidUsage("'as' bindings are not supported")
    call = item.context_expr
    assert isinstance(call, ast.Call)
    if call.args or call.keywords:
        raise exc.DeviceScopeInvalidUsage("it does not accept arguments")
    callee = call.func
    assert isinstance(callee, ast.Attribute)
    assert isinstance(node, ExtendedAST)
    with node._location:
        return create(
            ast.For,
            target=create(ast.Name, id="_", ctx=ast.Store()),
            iter=create(
                ast.Call,
                func=create(
                    ast.Attribute, value=callee.value, attr="grid", ctx=ast.Load()
                ),
                args=[create(ast.Constant, value=1, kind=None)],
                keywords=[],
            ),
            body=node.body,
            orelse=[],
            type_comment=None,
        )
