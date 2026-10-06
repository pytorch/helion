# Runtime support for the register-MMA GEMM family (``cute_matmul_family=
# "warp_mma"``, see ``cute_warp_mma_gemm.py``).
#
# NOTE: deliberately NO ``from __future__ import annotations`` (see
# _flash_runtime.py): these helpers are imported by the generated module and
# traced inline into the kernel body.  Every helper is one PTX instruction, so
# the emitted body is a straight line of SSA values.  The helpers shared with
# the flash row programs are re-exported from their runtime module.  The
# global loads carry a memory clobber (like the stores) so an aux row that
# aliases the output (an in-place residual) is read before it is written.
import cutlass
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T
from cutlass.cutlass_dsl import dsl_user_op

from ._flash_row_mma_runtime import _ASM_KW
from ._flash_row_mma_runtime import _floats
from ._flash_row_mma_runtime import _words
from ._flash_row_mma_runtime import cp_async_16  # noqa: F401
from ._flash_row_mma_runtime import ldmatrix_x4  # noqa: F401
from ._flash_row_mma_runtime import ldmatrix_x4_trans  # noqa: F401
from ._flash_row_mma_runtime import mma_bf16  # noqa: F401
from ._flash_row_mma_runtime import mma_f16  # noqa: F401
from ._flash_row_mma_runtime import pack_bf16x2  # noqa: F401
from ._flash_row_mma_runtime import pack_f16x2  # noqa: F401
from ._flash_row_mma_runtime import stg_f32  # noqa: F401


@dsl_user_op
def ldmatrix_x2(
    smem_address: object, *, loc: object = None, ip: object = None
) -> tuple[object, ...]:
    """``ldmatrix.sync.aligned.m8n8.x2.shared.b16``: two 8x8 b16 tiles whose
    row addresses lanes 0-15 supply."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 2),  # pyrefly: ignore[missing-attribute]
        [cutlass.Int32(smem_address).ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {$0, $1}, [$2];",
        "=r,=r,r,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return _words(result, 2, loc=loc, ip=ip)


@dsl_user_op
def ldmatrix_x2_trans(
    smem_address: object, *, loc: object = None, ip: object = None
) -> tuple[object, ...]:
    """``ldmatrix ... .x2.trans``: lane (g, c) receives rows 2c, 2c+1 of column g."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 2),  # pyrefly: ignore[missing-attribute]
        [cutlass.Int32(smem_address).ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {$0, $1}, [$2];",
        "=r,=r,r,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return _words(result, 2, loc=loc, ip=ip)


@dsl_user_op
def mma_e4m3(
    a0: object,
    a1: object,
    a2: object,
    a3: object,
    b0: object,
    b1: object,
    c0: object,
    c1: object,
    c2: object,
    c3: object,
    *,
    loc: object = None,
    ip: object = None,
) -> tuple[object, ...]:
    """``mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32``: lane (g, c)
    supplies four e4m3 of rows g / g+8 at k 4c..4c+3 (a0 / a1) and 16+4c..
    (a2 / a3), column g of B at the same k (b0 / b1)."""
    values = [
        *[cutlass.Uint32(v).ir_value(loc=loc, ip=ip) for v in (a0, a1, a2, a3, b0, b1)],
        *[cutlass.Float32(v).ir_value(loc=loc, ip=ip) for v in (c0, c1, c2, c3)],
    ]
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4),  # pyrefly: ignore[missing-attribute]
        values,
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return _floats(result, 4, loc=loc, ip=ip)


@dsl_user_op
def ldg_b32(
    gmem_address: object, *, loc: object = None, ip: object = None
) -> cutlass.Uint32:
    """``ld.global.b32``: one 4-byte word."""
    result = llvm.inline_asm(
        T.i32(),
        [cutlass.Int64(gmem_address).ir_value(loc=loc, ip=ip)],
        "ld.global.b32 $0, [$1];",
        "=r,l,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return cutlass.Uint32(result)


@dsl_user_op
def ldg_f32(
    gmem_address: object, *, loc: object = None, ip: object = None
) -> cutlass.Float32:
    """``ld.global.f32``: one fp32 value."""
    result = llvm.inline_asm(
        T.f32(),
        [cutlass.Int64(gmem_address).ir_value(loc=loc, ip=ip)],
        "ld.global.f32 $0, [$1];",
        "=f,l,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return cutlass.Float32(result)


@dsl_user_op
def ldg_v2_f32(
    gmem_address: object, *, loc: object = None, ip: object = None
) -> tuple[object, ...]:
    """``ld.global.v2.f32``: two consecutive fp32 values (8-byte aligned)."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 2),  # pyrefly: ignore[missing-attribute]
        [cutlass.Int64(gmem_address).ir_value(loc=loc, ip=ip)],
        "ld.global.v2.f32 {$0, $1}, [$2];",
        "=f,=f,l,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return _floats(result, 2, loc=loc, ip=ip)


def _ldg_b16_as_f32(
    kind: str, gmem_address: object, *, loc: object, ip: object
) -> cutlass.Float32:
    result = llvm.inline_asm(
        T.f32(),
        [cutlass.Int64(gmem_address).ir_value(loc=loc, ip=ip)],
        f"{{ .reg .b16 h; ld.global.b16 h, [$1]; cvt.f32.{kind} $0, h; }}",
        "=f,l,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return cutlass.Float32(result)


@dsl_user_op
def ldg_f16_as_f32(
    gmem_address: object, *, loc: object = None, ip: object = None
) -> cutlass.Float32:
    """One fp16 element loaded and widened to fp32 (exact)."""
    return _ldg_b16_as_f32("f16", gmem_address, loc=loc, ip=ip)


@dsl_user_op
def ldg_bf16_as_f32(
    gmem_address: object, *, loc: object = None, ip: object = None
) -> cutlass.Float32:
    """One bf16 element loaded and widened to fp32 (exact)."""
    return _ldg_b16_as_f32("bf16", gmem_address, loc=loc, ip=ip)


def _unpack_x2(
    kind: str, word: object, *, loc: object, ip: object
) -> tuple[object, ...]:
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 2),  # pyrefly: ignore[missing-attribute]
        [cutlass.Uint32(word).ir_value(loc=loc, ip=ip)],
        "{ .reg .b16 lo, hi; mov.b32 {lo, hi}, $2; "
        f"cvt.f32.{kind} $0, lo; cvt.f32.{kind} $1, hi; }}",
        "=f,=f,r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return _floats(result, 2, loc=loc, ip=ip)


@dsl_user_op
def unpack_f16x2(
    word: object, *, loc: object = None, ip: object = None
) -> tuple[object, ...]:
    """``(low, high)`` fp32 values of a packed f16x2 word (exact)."""
    return _unpack_x2("f16", word, loc=loc, ip=ip)


@dsl_user_op
def unpack_bf16x2(
    word: object, *, loc: object = None, ip: object = None
) -> tuple[object, ...]:
    """``(low, high)`` fp32 values of a packed bf16x2 word (exact)."""
    return _unpack_x2("bf16", word, loc=loc, ip=ip)


@dsl_user_op
def stg_b32(
    gmem_address: object, word: object, *, loc: object = None, ip: object = None
) -> None:
    """``st.global.b32``: one 4-byte word."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int64(gmem_address).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(word).ir_value(loc=loc, ip=ip),
        ],
        "st.global.b32 [$0], $1;",
        "l,r,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )


@dsl_user_op
def stg_v2_f32(
    gmem_address: object,
    x0: object,
    x1: object,
    *,
    loc: object = None,
    ip: object = None,
) -> None:
    """``st.global.v2.f32``: two consecutive fp32 values (8-byte aligned)."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int64(gmem_address).ir_value(loc=loc, ip=ip),
            cutlass.Float32(x0).ir_value(loc=loc, ip=ip),
            cutlass.Float32(x1).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v2.f32 [$0], {$1, $2};",
        "l,f,f,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )


def _stg_f32_as_b16(
    kind: str, gmem_address: object, value: object, *, loc: object, ip: object
) -> None:
    llvm.inline_asm(
        None,
        [
            cutlass.Int64(gmem_address).ir_value(loc=loc, ip=ip),
            cutlass.Float32(value).ir_value(loc=loc, ip=ip),
        ],
        f"{{ .reg .b16 h; cvt.rn.{kind}.f32 h, $1; st.global.b16 [$0], h; }}",
        "l,f,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )


@dsl_user_op
def stg_f16(
    gmem_address: object, value: object, *, loc: object = None, ip: object = None
) -> None:
    """One fp32 value rounded (RN) to fp16 and stored (2-byte aligned)."""
    _stg_f32_as_b16("f16", gmem_address, value, loc=loc, ip=ip)


@dsl_user_op
def stg_bf16(
    gmem_address: object, value: object, *, loc: object = None, ip: object = None
) -> None:
    """One fp32 value rounded (RN) to bf16 and stored (2-byte aligned)."""
    _stg_f32_as_b16("bf16", gmem_address, value, loc=loc, ip=ip)
