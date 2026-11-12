"""Fuse a float multiply into the add that consumes it, as ``metal::fma``.

PyTorch and MLX compile shaders in safe math mode, where Metal keeps every
``a * b + c`` a multiply and an add; Triton and nvcc contract them by default.
``#pragma clang fp contract(fast)`` would too, but Metal then also fuses
across an explicit narrowing (``float(half(a * b)) + c`` becomes one fma),
dropping a rounding the kernel asked for.  So the fusion happens here, on the
generated statements: a product of names or literals, used once, by an add in
the same block, with neither operand reassigned in between, and all three
operands inferred ``float`` (index math and narrower types keep their
operators).
"""

from __future__ import annotations

import ast
from collections import Counter
from typing import TYPE_CHECKING

from .lane_fold import tl_load

if TYPE_CHECKING:
    from collections.abc import Mapping

FLOAT = "float"
_SCALAR_TYPES = frozenset(
    {"float", "half", "bfloat", "int", "uint", "short", "long", "bool"}
)

#: Calls whose result has their first argument's type.
_SAME_TYPE_CALLS = frozenset(
    {
        "exp",
        "exp2",
        "log",
        "log2",
        "sqrt",
        "rsqrt",
        "abs",
        "fabs",
        "tanh",
        "sin",
        "cos",
        "max",
        "min",
        "fmax",
        "fmin",
        "floor",
        "ceil",
        "fma",
    }
)


def fuse_multiply_adds(
    stmts: list[ast.stmt], types: Mapping[str, str]
) -> list[ast.stmt]:
    """Rewrite ``t = a * b; ...; u = t + c`` (or ``c + t``) to
    ``u = metal::fma(a, b, c)`` throughout *stmts*, dropping ``t``.  *types*
    maps each buffer and scalar parameter to its MSL element type."""
    uses = Counter(
        node.id
        for stmt in stmts
        for node in ast.walk(stmt)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    )
    assigned = Counter(
        node.id
        for stmt in stmts
        for node in ast.walk(stmt)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    )
    return _fuse_block(stmts, uses, assigned, dict(types))


def _fuse_block(
    stmts: list[ast.stmt],
    uses: Counter[str],
    assigned: Counter[str],
    env: dict[str, str],
) -> list[ast.stmt]:
    products: dict[str, tuple[int, ast.BinOp]] = {}
    dropped: set[int] = set()
    for i, stmt in enumerate(stmts):
        if isinstance(stmt, (ast.For, ast.If)):
            stmt.body = _fuse_block(stmt.body, uses, assigned, env)
            stmt.orelse = _fuse_block(stmt.orelse, uses, assigned, env)
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target, value = stmt.targets[0], stmt.value
            if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
                for product, other in (
                    (value.left, value.right),
                    (value.right, value.left),
                ):
                    if (
                        isinstance(product, ast.Name)
                        and product.id in products
                        and _type(other, env) == FLOAT
                    ):
                        j, mul = products.pop(product.id)
                        stmt.value = ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="metal", ctx=ast.Load()),
                                attr="fma",
                                ctx=ast.Load(),
                            ),
                            args=[mul.left, mul.right, other],
                            keywords=[],
                        )
                        dropped.add(j)
                        break
            if isinstance(target, ast.Name):
                # MSL fixes a name's type where ``auto`` declares it; one
                # assigned values of two types has no one type here.
                typed = _type(stmt.value, env) or ""
                env[target.id] = typed if env.get(target.id, typed) == typed else ""
                if (
                    isinstance(value, ast.BinOp)
                    and isinstance(value.op, ast.Mult)
                    and all(
                        isinstance(x, (ast.Name, ast.Constant))
                        and _type(x, env) == FLOAT
                        for x in (value.left, value.right)
                    )
                    and uses[target.id] == 1
                    and assigned[target.id] == 1
                ):
                    products[target.id] = (i, value)
                    continue
        # A product whose operands this statement reassigns, or which it
        # reads or writes itself, can no longer move.
        written = _assigned_names(stmt)
        loaded = _loaded_names(stmt)
        for name, (_, mul) in list(products.items()):
            if written & _loaded_names(mul) or name in loaded | written:
                del products[name]
    return [stmt for i, stmt in enumerate(stmts) if i not in dropped]


def _type(node: ast.expr, env: Mapping[str, str]) -> str | None:
    """The MSL scalar type *node* evaluates to, where it is evident."""
    if isinstance(node, ast.Name):
        return env.get(node.id) or None
    if isinstance(node, ast.Constant):
        if isinstance(node.value, float):
            return FLOAT  # a float literal beside a float stays float
        return None
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        return env.get(node.value.id) or None
    if isinstance(node, ast.IfExp):
        body, orelse = _type(node.body, env), _type(node.orelse, env)
        return body if body == orelse else None
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
    ):
        left, right = _type(node.left, env), _type(node.right, env)
        return left if left == right else None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return _type(node.operand, env)
    if (load := tl_load(node)) is not None:
        # tl.load(buf + offset, ...): an element of buf, or the fill of a
        # masked-off one, converted to it.
        ptr = load.args[0]
        if isinstance(ptr, ast.BinOp) and isinstance(ptr.left, ast.Name):
            return env.get(ptr.left.id) or None
        return None
    if isinstance(node, ast.Call) and node.args:
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else None
        if isinstance(func, ast.Name):
            name = func.id
        first = node.args[0]
        if name in ("static_cast", "as_type") and isinstance(first, ast.Name):
            return first.id
        if isinstance(func, ast.Name) and name in _SCALAR_TYPES:  # float(0.0)
            return name
        if name in _SAME_TYPE_CALLS:
            return _type(first, env)
    return None


def _assigned_names(stmt: ast.stmt) -> set[str]:
    return {
        n.id
        for n in ast.walk(stmt)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }


def _loaded_names(node: ast.AST) -> set[str]:
    return {
        n.id
        for n in ast.walk(node)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
