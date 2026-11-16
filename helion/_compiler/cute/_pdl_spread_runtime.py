# Runtime support for ``cute_pdl_spread`` (see ``pdl_release.py``): the pad
# allocation that holds a programmatic dependent's CTAs to one wave's density
# has no reader, and the DSL drops a shared allocation nothing uses, so the
# generated kernel hands its address to this empty, side-effecting inline
# asm.  Imported by the generated module and traced inline into the body.
# (Unlike _flash_runtime.py there is no ``@cute.struct`` here, so the
# stringified annotations of the future import are harmless.)
from __future__ import annotations

import cutlass
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from .device.ptx import ASM_KW


@dsl_user_op
def keep_smem_live(
    smem_address: object, *, loc: object = None, ip: object = None
) -> None:
    """Keep the shared allocation at ``smem_address`` live: an inline asm
    with side effects that reads the address and emits no instruction."""
    llvm.inline_asm(
        None,
        [cutlass.Int32(smem_address).ir_value(loc=loc, ip=ip)],
        "",
        "r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **ASM_KW,
    )
