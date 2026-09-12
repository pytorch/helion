"""FlyDSL-backend codegen for ops defined in ``helion.language.reduce_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "flydsl")``
registrations; ``_codegen_modules`` imports it so registration keeps the same
eager timing as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ... import exc
from ...language import _decorators
from ...language.reduce_ops import _reduce

if TYPE_CHECKING:
    import ast

    from ..inductor_lowering import CodegenState


@_decorators.codegen(_reduce, "flydsl")
def _(state: CodegenState) -> ast.AST | list[ast.AST]:
    # ``hl.reduce`` (the explicit user-combine primitive) is not yet supported on
    # flydsl. The whole-row reductions the flydsl kernels actually use --
    # ``torch.sum`` / ``mean`` / ``amax`` / ``amin`` etc. -- lower through the
    # reduction *strategy* path (see FlyDSLBackend.reduction_expr), which folds
    # each thread's per-lane vector to a scalar before the cross-lane warp
    # shuffle. This ``_reduce`` codegen would instead shuffle the per-thread
    # vector directly and store a vector into a scalar slot, which crashes the
    # AMDGPU backend ("cannot scalarize buffer.store operand"). Reject it cleanly
    # until the vectorized per-lane fold is implemented; the standard PyTorch
    # reductions work today and are the documented path.
    raise exc.BackendUnsupported(
        "flydsl",
        "hl.reduce (use torch.sum/mean/amax/amin, which lower through the "
        "supported reduction path)",
    )
