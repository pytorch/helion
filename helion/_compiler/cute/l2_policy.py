# pyrefly: ignore-errors
"""L2 cache-policy load helpers for CuTe codegen.

``cute.arch.load`` exposes L1 eviction priorities and the ``.cs`` cache
operator, but not L2 policy-descriptor hints.  Triton's ``evict_last``
lowers to ``createpolicy.fractional.L2::evict_last`` + a cache-hint load,
which keeps up to ~L2-size of a streaming input resident across other
traffic (including do_bench's flush) — measured worth ~1.7% on an fp32
elementwise mul on B200.  This emits the same PTX via inline asm.

Called during ``@cute.kernel`` tracing (plain Python building MLIR),
like ``vec_utils``.
"""

from __future__ import annotations

import cutlass
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import vector as _vector_dialect
from cutlass.cutlass_dsl import dsl_user_op

_ASM_V4_B32_L2_EVICT_LAST = (
    "{\n"
    ".reg .b64 pol;\n"
    "createpolicy.fractional.L2::evict_last.b64 pol, 1.0;\n"
    "ld.global.L2::cache_hint.v4.b32 {$0,$1,$2,$3}, [$4], pol;\n"
    "}"
)


@dsl_user_op
def load_v16b_l2_evict_last(
    ptr: object,
    vec_type: ir.VectorType,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> ir.Value:
    """16-byte vector load with an ``L2::evict_last`` cache-hint policy.

    Returns a raw ``ir.Value`` of ``vec_type`` (any 16-byte vector shape),
    matching what ``cute.arch.load`` returns for vector dtypes.
    """
    addr = ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    u32 = cutlass.Uint32.mlir_type
    res_ty = llvm.StructType.get_literal([u32] * 4)
    res = llvm.inline_asm(
        res_ty,
        [addr],
        _ASM_V4_B32_L2_EVICT_LAST,
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    vals = [
        llvm.extractvalue(u32, res, [i], loc=loc, ip=ip)  # pyrefly: ignore
        for i in range(4)
    ]
    v4_ty = ir.VectorType.get([4], u32)
    v4 = _vector_dialect.from_elements(v4_ty, vals, loc=loc, ip=ip)
    if str(vec_type) == str(v4_ty):
        return v4
    return _vector_dialect.bitcast(vec_type, v4, loc=loc, ip=ip)
