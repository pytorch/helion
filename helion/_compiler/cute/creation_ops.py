"""CuTe-backend codegen for ops defined in ``helion.language.creation_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module); ``_codegen_modules`` imports this module so its
``@_decorators.codegen(op, "cute")`` registrations run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...language import _decorators
from ...language.creation_ops import full

if TYPE_CHECKING:
    import ast

    from ..inductor_lowering import CodegenState


@_decorators.codegen(full, "cute")
def _full_codegen_cute(state: CodegenState) -> ast.AST:
    """``hl.full`` with one element per thread: its value, converted to the
    float dtype as the pointwise lowering converts an integer or a bool
    (``integer_to_float_expr``), where MLIR could fold an 8-bit float
    conversion into the integer's."""
    from .handler_ops import integer_to_float_expr
    from .handler_ops import scalar_value_dtype

    fake_value = state.fake_value
    value = state.proxy_arg(1)
    if isinstance(fake_value, torch.Tensor) and not isinstance(
        value, (int, float, bool)
    ):
        converted = integer_to_float_expr(
            state.ast_arg(1),
            fake_value.dtype,
            scalar_value_dtype(value),
            state.device_function,
        )
        if converted is not None:
            return converted
    return full._codegen["common"](state)  # pyrefly: ignore[missing-attribute]
