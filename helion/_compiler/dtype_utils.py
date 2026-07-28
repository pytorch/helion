from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .compile_environment import CompileEnvironment

if TYPE_CHECKING:
    import ast


def cast_ast(
    x: ast.AST,
    dtype: torch.dtype,
    source_dtype: torch.dtype | None = None,
) -> ast.AST:
    """Return an AST that casts expression `x` to the backend dtype string."""
    env = CompileEnvironment.current()
    return env.backend.cast_ast(x, dtype, source_dtype)


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
    lhs_out = lhs if lhs_dtype == common else cast_ast(lhs, common, lhs_dtype)
    rhs_out = rhs if rhs_dtype == common else cast_ast(rhs, common, rhs_dtype)
    return lhs_out, rhs_out, common
