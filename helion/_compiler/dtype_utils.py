from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch._inductor.utils import triton_type

from .ast_extension import expr_from_string

if TYPE_CHECKING:
    import ast


def cast_ast(x: ast.AST, dtype: torch.dtype) -> ast.AST:
    """Return an AST that casts expression `x` to Triton `dtype` via tl.cast."""

    return expr_from_string(f"tl.cast({{x}}, {triton_type(dtype)})", x=x)


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


def emit_tl_dot(
    lhs: ast.AST,
    rhs: ast.AST,
    *,
    input_precision: str | None = None,
    acc: ast.AST | None = None,
    out_dtype: torch.dtype | None = None,
) -> ast.AST:
    """Build a tl.dot AST with optional acc/input_precision/out_dtype.

    The caller is responsible for ensuring compatible operand/accumulator
    dtypes for fused accumulation when providing `acc`.
    """

    parts = ["tl.dot({lhs}, {rhs}"]
    if acc is not None:
        parts.append(", acc={acc}")
    if input_precision is not None:
        parts.append(f", input_precision='{input_precision}'")
    if out_dtype is not None:
        parts.append(f", out_dtype={triton_type(out_dtype)}")
    parts.append(")")
    tpl = "".join(parts)
    kwargs: dict[str, ast.AST] = {"lhs": lhs, "rhs": rhs}
    if acc is not None:
        kwargs["acc"] = acc
    return expr_from_string(tpl, **kwargs)
