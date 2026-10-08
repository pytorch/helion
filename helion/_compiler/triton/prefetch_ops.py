"""Triton-backend codegen for ``hl.prefetch``."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast

import torch

from ...language import _decorators
from ...language.prefetch_ops import prefetch
from ...language.prefetch_ops import prefetch_nbytes
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState

# ptxas caps a bulk prefetch's size operand at 1 MiB - 16 bytes.
_MAX_BYTES = 1048560


@_decorators.codegen(prefetch, "triton")
def _(state: CodegenState) -> ast.AST:
    tensor = state.proxy_arg(0)
    index = state.proxy_arg(1)
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(index, (list, tuple))
    nbytes = prefetch_nbytes(tensor, [*index])
    terms = []
    placeholders: dict[str, ast.AST] = {}
    for position, value in enumerate(cast("list[object]", state.ast_args[1])):
        if not isinstance(value, int):
            assert isinstance(value, ast.AST)
            placeholders[name := f"prefetch_index{position}"] = value
            value = f"{{{name}}}"
        stride = state.device_function.tensor_stride(tensor, position).name
        terms.append(f"({value}) * {stride}")
    offset = " + ".join(terms) or "0"
    base = f"{state.device_function.tensor_arg(tensor).name} + ({offset})"
    for begin in range(0, nbytes, _MAX_BYTES):
        size = min(_MAX_BYTES, nbytes - begin)
        state.codegen.add_statement(
            statement_from_string(
                f"helion_cache_hints.prefetch_l2({base}, {begin}, {size})",
                **placeholders,
            )
        )
    return expr_from_string("None")
