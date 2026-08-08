"""Pallas-backend codegen for ``helion.language.barrier``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...language import _decorators
from ...language.barrier import barrier
from ..ast_extension import expr_from_string

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState


@_decorators.codegen(barrier, "pallas")
def _(state: CodegenState) -> object:
    # Pallas implements grid barriers as sequential host launches.
    return expr_from_string("None")
