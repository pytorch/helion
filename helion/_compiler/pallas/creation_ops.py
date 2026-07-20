"""Pallas-backend codegen for ops defined in ``helion.language.creation_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "pallas")``
registrations; ``creation_ops`` imports it at the bottom so registration keeps
the same eager timing as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...language import _decorators
from ...language.creation_ops import full

if TYPE_CHECKING:
    import ast

    from ..inductor_lowering import CodegenState


@_decorators.codegen(full, "pallas")
def _full_codegen_pallas(state: CodegenState) -> ast.AST:
    """Pallas codegen for hl.full / hl.zeros.

    Always lowers to a plain ``jnp.full`` bound to a fresh local, regardless
    of pallas_loop_type.  The previous emit_pipeline/fori_loop path returned
    a bare scratch ref AST, which broke any downstream arithmetic on the
    result outside the inner loop -- e.g. ``acc = hl.zeros(...); acc += x``
    emitted ``scratch + x`` and JAX raised
    ``'AbstractRef' object has no attribute '_add'`` at trace time
    (``Refs`` don't support arithmetic; only ``ref[...]`` reads).

    When the result is loop-carried, ``_setup_loop_carried_state`` allocates
    a scratch buffer at the loop boundary and copies the init value in --
    no scratch needed at the ``hl.zeros`` site itself.
    """
    return full._codegen["common"](state)  # pyrefly: ignore[missing-attribute]
