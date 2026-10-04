"""Pallas-backend codegen for the barrier op defined in ``helion.language.barrier``.

A Pallas kernel with a barrier has several top-level loops, so it lowers to a
sequential-roots program whose program order already orders every phase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...language import _decorators
from ...language.barrier import barrier
from ..ast_extension import expr_from_string

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState


@_decorators.codegen(barrier, "pallas")
def _(state: CodegenState) -> object:
    return expr_from_string("None")
