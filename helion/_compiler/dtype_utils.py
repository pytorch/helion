from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .ast_extension import expr_from_string
from .compile_environment import CompileEnvironment

if TYPE_CHECKING:
    import ast


def cast_ast(x: ast.AST, dtype: torch.dtype) -> ast.AST:
    """Return an AST that casts expression `x` to the backend dtype string."""
    env = CompileEnvironment.current()
    dtype_str = env.backend.dtype_str(dtype)
    cast_str = env.backend.cast_expr("{x}", dtype_str)
    return expr_from_string(cast_str, x=x)


def promote_and_cast_pair(
    lhs: ast.AST,
    rhs: ast.AST,
    lhs_dtype: torch.dtype,
    rhs_dtype: torch.dtype,
) -> tuple[ast.AST, ast.AST, torch.dtype]:
    """Cast `lhs` and `rhs` to a common promoted dtype when needed.

    Returns (lhs_cast, rhs_cast, common_dtype). If dtypes already match, the
    original ASTs are returned unchanged to avoid redundant casts.
    """

    common = torch.promote_types(lhs_dtype, rhs_dtype)
    lhs_out = lhs if lhs_dtype == common else cast_ast(lhs, common)
    rhs_out = rhs if rhs_dtype == common else cast_ast(rhs, common)
    return lhs_out, rhs_out, common
