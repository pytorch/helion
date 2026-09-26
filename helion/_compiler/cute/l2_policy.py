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
import torch

from . import cutedsl_compat
from .cutedsl_compat import L2_EVICT_LAST_STORE_ABI_VERSION

__all__ = [
    "L2_EVICT_LAST_STORE_ABI_VERSION",
    "fixed_l2_evict_last_store_policy_supported",
]

_ASM_V4_B32_L2_EVICT_LAST = (
    "{\n"
    ".reg .b64 pol;\n"
    "createpolicy.fractional.L2::evict_last.b64 pol, 1.0;\n"
    "ld.global.L2::cache_hint.v4.b32 {$0,$1,$2,$3}, [$4], pol;\n"
    "}"
)

_ASM_STORE_V4_B32_L2_EVICT_LAST = (
    "st.global.L2::cache_hint.v4.u32 [$0], {$1,$2,$3,$4}, $5;"
)
_L2_EVICT_LAST_POLICY = 0x14F0000000000000


def fixed_l2_evict_last_store_policy_supported(
    target_device_capability: tuple[int, int] | None,
) -> bool:
    """Whether the validated opaque store-policy descriptor is safe to emit.

    The descriptor was obtained from CUDA 13 for SM103.  It is intentionally
    not treated as a portable PTX constant; other targets/toolchains fall back
    by making the source transform ineligible.
    """
    return cutedsl_compat.fixed_l2_evict_last_store_policy_supported(
        target_device_capability,
        torch.version.cuda,
    )


@dsl_user_op
def _load_vector_with_policy(
    ptr: object,
    vec_type: ir.VectorType,
    assembly: str,
    *,
    words: int = 4,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> ir.Value:
    """Load aligned words and bitcast to the requested vector type."""
    addr = ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    u32 = cutlass.Uint32.mlir_type
    res_ty = llvm.StructType.get_literal([u32] * words)
    res = llvm.inline_asm(
        res_ty,
        [addr],
        assembly,
        ",".join(["=r"] * words + ["l"]),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    vals = [
        llvm.extractvalue(u32, res, [i], loc=loc, ip=ip)  # pyrefly: ignore
        for i in range(words)
    ]
    v4_ty = ir.VectorType.get([words], u32)
    v4 = _vector_dialect.from_elements(v4_ty, vals, loc=loc, ip=ip)
    if str(vec_type) == str(v4_ty):
        return v4
    return _vector_dialect.bitcast(vec_type, v4, loc=loc, ip=ip)


@dsl_user_op
def load_v16b_l2_evict_last(
    ptr: object,
    vec_type: ir.VectorType,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> ir.Value:
    """16-byte vector load with an L2 retention hint."""
    return _load_vector_with_policy(
        ptr, vec_type, _ASM_V4_B32_L2_EVICT_LAST, loc=loc, ip=ip
    )


def _two_level_assembly(priority: str, words: int = 4) -> str:
    registers = ",".join(f"${index}" for index in range(words))
    return (
        "{\n"
        ".reg .b64 pol;\n"
        f"createpolicy.fractional.L2::{priority}.b64 pol, 1.0;\n"
        f"ld.global.L1::{priority}.L2::cache_hint.v{words}.b32 "
        f"{{{registers}}}, [${words}], pol;\n"
        "}"
    )


@dsl_user_op
def load_v16b_l1_l2_evict_first(
    ptr: object,
    vec_type: ir.VectorType,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> ir.Value:
    """16-byte vector load with matching L1 and L2 eviction hints."""
    return _load_vector_with_policy(
        ptr, vec_type, _two_level_assembly("evict_first"), loc=loc, ip=ip
    )


@dsl_user_op
def load_v16b_l1_l2_evict_last(
    ptr: object,
    vec_type: ir.VectorType,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> ir.Value:
    """16-byte vector load with matching L1 and L2 retention hints."""
    return _load_vector_with_policy(
        ptr, vec_type, _two_level_assembly("evict_last"), loc=loc, ip=ip
    )


def _store_u32x4_l2_evict_last(ptr: object, values: list[object]) -> None:
    """Store four packed words with the CUDA L2 ``evict_last`` policy."""

    assert len(values) == 4
    address = ptr.toint().ir_value()
    words = [cutlass.Uint32(value).ir_value() for value in values]
    policy = cutlass.Uint64(_L2_EVICT_LAST_POLICY).ir_value()
    llvm.inline_asm(
        None,
        [address, *words, policy],
        _ASM_STORE_V4_B32_L2_EVICT_LAST,
        "l,r,r,r,r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def store_u16x8_l2_evict_last(ptr: object, values: list[object]) -> None:
    """Store one 16-byte Uint16 fragment with an L2 retention hint."""

    assert len(values) == 8
    words = [
        cutlass.Uint32(values[index])
        | (cutlass.Uint32(values[index + 1]) << cutlass.Uint32(16))
        for index in range(0, 8, 2)
    ]
    _store_u32x4_l2_evict_last(ptr, words)


def store_u32x4_l2_evict_last(ptr: object, values: list[object]) -> None:
    """Store one 16-byte Uint32 fragment with an L2 retention hint."""

    _store_u32x4_l2_evict_last(ptr, values)


@dsl_user_op
def load_v8b_l1_l2_evict_first(
    ptr: object,
    vec_type: ir.VectorType,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> ir.Value:
    """8-byte vector load with matching L1 and L2 eviction hints."""
    return _load_vector_with_policy(
        ptr, vec_type, _two_level_assembly("evict_first", 2), words=2, loc=loc, ip=ip
    )


@dsl_user_op
def load_v8b_l1_l2_evict_last(
    ptr: object,
    vec_type: ir.VectorType,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> ir.Value:
    """8-byte vector load with matching L1 and L2 eviction hints."""
    return _load_vector_with_policy(
        ptr, vec_type, _two_level_assembly("evict_last", 2), words=2, loc=loc, ip=ip
    )
