"""Lower hinted scalar loads after all CuTe memory scheduling passes."""

from __future__ import annotations

import ast
from typing import cast

_SCALAR_TYPES = frozenset(("Float32", "Float64", "Int32", "Int64", "Uint32", "Uint64"))
_POLICIES = {
    ("level1_eviction_priority", "evict_first"): "first",
    ("level1_eviction_priority", "evict_last"): "last",
    ("cop", "cs"): "streaming",
}


class _LowerScalarPolicyLoads(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if (
            not isinstance(node.func, ast.Attribute)
            or node.func.attr != "load"
            or not isinstance(node.func.value, ast.Attribute)
            or node.func.value.attr != "arch"
            or not isinstance(node.func.value.value, ast.Name)
            or node.func.value.value.id != "cute"
            or len(node.args) != 2
            or len(node.keywords) != 1
        ):
            return node
        dtype = node.args[1]
        keyword = node.keywords[0]
        if (
            not isinstance(dtype, ast.Attribute)
            or not isinstance(dtype.value, ast.Name)
            or dtype.value.id != "cutlass"
            or dtype.attr not in _SCALAR_TYPES
            or keyword.arg is None
            or not isinstance(keyword.value, ast.Constant)
            or not isinstance(keyword.value.value, str)
        ):
            return node
        policy = _POLICIES.get((keyword.arg, keyword.value.value))
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
    lower = _LowerScalarPolicyLoads()
    return [cast("ast.stmt", lower.visit(statement)) for statement in body]
