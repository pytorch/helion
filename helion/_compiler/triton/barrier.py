"""Triton-backend codegen for the barrier op defined in ``helion.language.barrier``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "triton")``
registrations; ``barrier`` imports it at the bottom so registration keeps the
same eager timing as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...language import _decorators
from ...language.barrier import barrier
from ..ast_extension import expr_from_string

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState


@_decorators.codegen(barrier, "triton")
def _(state: CodegenState) -> object:
    # No device code emitted; barrier only affects host-side scheduling.
    return expr_from_string("None")
