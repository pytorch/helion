"""Lower hinted scalar loads after all CuTe memory scheduling passes."""

from __future__ import annotations

import ast
from typing import cast

from .cache_policy_loads import LOAD_POLICY_KEYWORDS
from .cache_policy_loads import CachePolicy

_SCALAR_TYPES = frozenset(("Float32", "Float64", "Int32", "Int64", "Uint32", "Uint64"))


def _is_arch_load(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "load"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "arch"
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "cute"
    )


def scalar_load_policy(node: ast.Call) -> str | None:
    """The ``_cute_scalar_policy_load`` policy a hinted scalar
    ``cute.arch.load(ptr, cutlass.<T>, <hint>)`` lowers to, or None when
    ``lower_scalar_policy_loads`` leaves the call as it is."""
    if not _is_arch_load(node) or len(node.args) != 2 or len(node.keywords) != 1:
        return None
    dtype = node.args[1]
    if (
        not isinstance(dtype, ast.Attribute)
        or not isinstance(dtype.value, ast.Name)
        or dtype.value.id != "cutlass"
        or dtype.attr not in _SCALAR_TYPES
    ):
        return None
    cache_policy = CachePolicy.from_load_keywords(node.keywords)
    return cache_policy.scalar_policy if cache_policy is not None else None


def is_extended_scalar_load(node: ast.Call) -> bool:
    """Whether a hinted scalar ``cute.arch.load`` reaches NVVM as an extended
    load: one with cache keywords and a scalar dtype (not a vector type) that
    ``lower_scalar_policy_loads`` does not turn into exact-policy PTX."""
    if not _is_arch_load(node) or not node.keywords:
        return False
    if len(node.args) >= 2 and (
        isinstance(node.args[1], ast.Call)
        and ast.unparse(node.args[1].func) == "ir.VectorType.get"
    ):
        return False
    return scalar_load_policy(node) is None


class _LowerScalarPolicyLoads(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        policy = scalar_load_policy(node)
        if policy is None:
            return node
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id="_cute_scalar_policy_load", ctx=ast.Load()),
                args=[*node.args, ast.Constant(value=policy)],
                keywords=[],
            ),
            node,
        )


def lower_scalar_policy_loads(body: list[ast.stmt]) -> list[ast.stmt]:
    """Preserve load recognition until the final device emission boundary.

    Mixing scalar NVVM extended loads with ordinary LLVM loads can abort the
    SDK load/store vectorizer. Exact-policy PTX avoids that grouping without
    changing the selected cache hints. Earlier scheduling and alias passes
    must still see their original ``cute.arch.load`` nodes.
    """
    if not any(
        isinstance(node, ast.keyword) and node.arg in LOAD_POLICY_KEYWORDS
        for statement in body
        for node in ast.walk(statement)
    ):
        # Every lowered load carries a policy keyword; without one the
        # transformer would return each statement unchanged.
        return list(body)
    lower = _LowerScalarPolicyLoads()
    return [cast("ast.stmt", lower.visit(statement)) for statement in body]
