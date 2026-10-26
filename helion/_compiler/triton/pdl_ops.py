"""Triton-backend codegen for the PDL ops defined in ``helion.language.pdl_ops``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...language import _decorators
from ...language.pdl_ops import pdl_launch_dependents
from ...language.pdl_ops import pdl_wait
from ..ast_extension import expr_from_string

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState


def _host_noop(state: CodegenState) -> object:
    # DeviceFunction emits the kernel entry/exit instructions.
    return expr_from_string("None")


_decorators.codegen(pdl_wait, "triton")(_host_noop)
_decorators.codegen(pdl_launch_dependents, "triton")(_host_noop)
