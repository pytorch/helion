# Runtime support for the ``cute_mma_dag`` warp-MMA dataflow owner (see
# ``cute_mma_dag.py`` / ``cute_mma_dag_codegen.py``).
#
# These helpers are imported by the generated module and traced inline into
# the kernel body.  Every helper is one PTX instruction, so the emitted body
# is a straight line of SSA values that ptxas cannot reorder around the
# cp.async wait that guards the operand data.  The MMA, ldmatrix, cp.async,
# pack and global load / store helpers are the row_mma / warp_mma ones,
# re-exported so the generated code names one runtime module.  (Unlike
# _flash_runtime.py there is no ``@cute.struct`` here, so the stringified
# annotations of the future import are harmless.)
from __future__ import annotations

import cutlass
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T
from cutlass.cutlass_dsl import dsl_user_op

from ._flash_row_mma_runtime import _ASM_KW
from ._flash_row_mma_runtime import _floats
from ._flash_row_mma_runtime import _words
from ._flash_row_mma_runtime import cp_async_16  # noqa: F401
from ._flash_row_mma_runtime import lane_group_sum  # noqa: F401
from ._flash_row_mma_runtime import ldmatrix_x4  # noqa: F401
from ._flash_row_mma_runtime import ldmatrix_x4_trans  # noqa: F401
from ._flash_row_mma_runtime import lds_f32  # noqa: F401
from ._flash_row_mma_runtime import lds_v4_f32  # noqa: F401
from ._flash_row_mma_runtime import mma_bf16  # noqa: F401
from ._flash_row_mma_runtime import mma_f16  # noqa: F401
from ._flash_row_mma_runtime import movmatrix_trans  # noqa: F401
from ._flash_row_mma_runtime import octet_sum  # noqa: F401
from ._flash_row_mma_runtime import pack_bf16x2  # noqa: F401
from ._flash_row_mma_runtime import pack_f16x2  # noqa: F401
from ._flash_row_mma_runtime import stg_f32  # noqa: F401
from ._flash_row_mma_runtime import stg_v4_b32  # noqa: F401
from ._flash_row_mma_runtime import sts_f32  # noqa: F401
from ._warp_mma_runtime import ldg_b32  # noqa: F401
from ._warp_mma_runtime import ldg_bf16_as_f32  # noqa: F401
from ._warp_mma_runtime import ldg_f16_as_f32  # noqa: F401
from ._warp_mma_runtime import ldg_f32  # noqa: F401
from ._warp_mma_runtime import ldg_v2_f32  # noqa: F401
from ._warp_mma_runtime import ldmatrix_x2  # noqa: F401
from ._warp_mma_runtime import ldmatrix_x2_trans  # noqa: F401
from ._warp_mma_runtime import mma_tf32  # noqa: F401
from ._warp_mma_runtime import stg_b32  # noqa: F401
from ._warp_mma_runtime import stg_bf16  # noqa: F401
from ._warp_mma_runtime import stg_f16  # noqa: F401
from ._warp_mma_runtime import stg_v2_f32  # noqa: F401
from ._warp_mma_runtime import unpack_bf16x2  # noqa: F401
from ._warp_mma_runtime import unpack_f16x2  # noqa: F401


@dsl_user_op
def sts_b32(
    smem_address: object, word: object, *, loc: object = None, ip: object = None
) -> None:
    """``st.shared.b32``: one 4-byte word (a packed 16-bit element pair of a
    cooperatively computed operand tile)."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(smem_address).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(word).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.b32 [$0], $1;",
        "r,r,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )


@dsl_user_op
def ldg_s32(
    gmem_address: object, *, loc: object = None, ip: object = None
) -> cutlass.Int32:
    """``ld.global.s32``: one int32 value (a side vector of window bounds)."""
    result = llvm.inline_asm(
        T.i32(),
        [cutlass.Int64(gmem_address).ir_value(loc=loc, ip=ip)],
        "ld.global.s32 $0, [$1];",
        "=r,l,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return cutlass.Int32(result)


@dsl_user_op
def f32_bits(value: object, *, loc: object = None, ip: object = None) -> object:
    """``mov.b32``: the bit pattern of an fp32 value (an operand word the
    tensor core truncates to tf32)."""
    result = llvm.inline_asm(
        T.i32(),
        [cutlass.Float32(value).ir_value(loc=loc, ip=ip)],
        "mov.b32 $0, $1;",
        "=r,f",
        has_side_effects=False,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return cutlass.Uint32(result)


@dsl_user_op
def shfl_pick_b32(
    lane: object,
    flag: object,
    x: object,
    y: object,
    *,
    loc: object = None,
    ip: object = None,
) -> object:
    """``flag ? y : x`` read from lane ``lane`` (two ``shfl.sync.idx`` and a
    ``selp``): one fp32 operand word picked from a producer fragment's
    register pair in another lane of the warp.  Every lane must take part."""
    result = llvm.inline_asm(
        T.i32(),
        [
            cutlass.Int32(lane).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(x).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(y).ir_value(loc=loc, ip=ip),
            cutlass.Int32(flag).ir_value(loc=loc, ip=ip),
        ],
        "{.reg .b32 t0, t1; .reg .pred p; "
        "shfl.sync.idx.b32 t0, $2, $1, 0x1f, 0xffffffff; "
        "shfl.sync.idx.b32 t1, $3, $1, 0x1f, 0xffffffff; "
        "setp.ne.s32 p, $4, 0; selp.b32 $0, t1, t0, p;}",
        "=r,r,r,r,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return cutlass.Uint32(result)


@dsl_user_op
def lds_b32(smem_address: object, *, loc: object = None, ip: object = None) -> object:
    """``ld.shared.b32``: one 4-byte word (a packed 16-bit element pair of a
    staged tile read at the lane's fragment coordinates)."""
    result = llvm.inline_asm(
        T.i32(),
        [cutlass.Int32(smem_address).ir_value(loc=loc, ip=ip)],
        "ld.shared.b32 $0, [$1];",
        "=r,r,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return cutlass.Uint32(result)


@dsl_user_op
def lds_v2_f32(
    smem_address: object, *, loc: object = None, ip: object = None
) -> tuple[object, ...]:
    """``ld.shared.v2.f32``: two consecutive fp32 values (8-byte aligned; an
    fp32 tile pair read at the lane's fragment coordinates)."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 2),  # pyrefly: ignore[missing-attribute]
        [cutlass.Int32(smem_address).ir_value(loc=loc, ip=ip)],
        "ld.shared.v2.f32 {$0, $1}, [$2];",
        "=f,=f,r,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return _floats(result, 2, loc=loc, ip=ip)


@dsl_user_op
def lds_v4_b32(
    smem_address: object, *, loc: object = None, ip: object = None
) -> tuple[object, ...]:
    """``ld.shared.v4.b32``: one 16-byte packet (eight 16-bit elements)."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),  # pyrefly: ignore[missing-attribute]
        [cutlass.Int32(smem_address).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return _words(result, 4, loc=loc, ip=ip)


@dsl_user_op
def sts_v4_b32(
    smem_address: object,
    w0: object,
    w1: object,
    w2: object,
    w3: object,
    *,
    loc: object = None,
    ip: object = None,
) -> None:
    """``st.shared.v4.b32``: one 16-byte packet."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(smem_address).ir_value(loc=loc, ip=ip),
            *[cutlass.Uint32(w).ir_value(loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
        ],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )


@dsl_user_op
def sts_v4_f32(
    smem_address: object,
    x0: object,
    x1: object,
    x2: object,
    x3: object,
    *,
    loc: object = None,
    ip: object = None,
) -> None:
    """``st.shared.v4.f32``: one 16-byte packet of four fp32 values (a
    widened tf32 operand row chunk)."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(smem_address).ir_value(loc=loc, ip=ip),
            *[cutlass.Float32(x).ir_value(loc=loc, ip=ip) for x in (x0, x1, x2, x3)],
        ],
        "st.shared.v4.f32 [$0], {$1, $2, $3, $4};",
        "r,f,f,f,f,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )


@dsl_user_op
def sts_v2_f32(
    smem_address: object,
    x0: object,
    x1: object,
    *,
    loc: object = None,
    ip: object = None,
) -> None:
    """``st.shared.v2.f32``: two consecutive fp32 values (8-byte aligned)."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(smem_address).ir_value(loc=loc, ip=ip),
            cutlass.Float32(x0).ir_value(loc=loc, ip=ip),
            cutlass.Float32(x1).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.v2.f32 [$0], {$1, $2};",
        "r,f,f,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )


@dsl_user_op
def sts_v2_b32(
    smem_address: object,
    w0: object,
    w1: object,
    *,
    loc: object = None,
    ip: object = None,
) -> None:
    """``st.shared.v2.b32``: two packed 16-bit pairs (8-byte aligned)."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(smem_address).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(w0).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(w1).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.v2.b32 [$0], {$1, $2};",
        "r,r,r,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
