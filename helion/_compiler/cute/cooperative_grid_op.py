"""Emit the installed CUDA toolkit's closed cooperative-groups device operation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

if TYPE_CHECKING:
    from ._mlir_compat import ir


@dsl_user_op
def toolkit_grid_sync(
    asm: str, *, loc: ir.Location | None = None, ip: ir.InsertionPoint | None = None
) -> None:
    llvm.inline_asm(
        None,
        [],
        asm,
        "~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
