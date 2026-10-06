# Runtime support for the register-MMA row-program flash family (``row_mma``).
#
# NOTE: deliberately NO ``from __future__ import annotations`` (see
# _flash_runtime.py): these helpers are imported by the generated module and
# traced inline into the kernel body.  Every helper is one PTX instruction (or
# a short warp-shuffle chain) with explicit register operands, so the emitted
# body is a straight line of SSA values that ptxas cannot reorder around the
# cp.async wait that guards the operand data.
import cutlass
from cutlass._mlir.dialects import llvm
import cutlass.cute as cute
from cutlass.cutlass_dsl import T
from cutlass.cutlass_dsl import dsl_user_op

_ASM_KW = {
    "is_align_stack": False,
    "asm_dialect": llvm.AsmDialect.AD_ATT,
}


def _words(
    result: object, count: int, *, loc: object, ip: object
) -> tuple[object, ...]:
    return tuple(
        cutlass.Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip))
        for i in range(count)
    )


def _floats(
    result: object, count: int, *, loc: object, ip: object
) -> tuple[object, ...]:
    return tuple(
        cutlass.Float32(llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip))
        for i in range(count)
    )


@dsl_user_op
def cp_async_16(
    smem_address: object, gmem_address: object, *, loc: object = None, ip: object = None
) -> None:
    """``cp.async.cg.shared.global`` of one 16-byte packet (L2 only)."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(smem_address).ir_value(loc=loc, ip=ip),
            cutlass.Int64(gmem_address).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.cg.shared.global [$0], [$1], 16;",
        "r,l,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )


@dsl_user_op
def ldmatrix_x4(
    smem_address: object, *, loc: object = None, ip: object = None
) -> tuple[object, ...]:
    """``ldmatrix.sync.aligned.m8n8.x4.shared.b16``: four 8x8 b16 tiles whose
    row addresses the lanes supply (lane l -> row l % 8 of tile l // 8); lane
    (g, c) receives row g, columns 2c, 2c+1 of each tile."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),  # pyrefly: ignore[missing-attribute]
        [cutlass.Int32(smem_address).ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return _words(result, 4, loc=loc, ip=ip)


@dsl_user_op
def ldmatrix_x4_trans(
    smem_address: object, *, loc: object = None, ip: object = None
) -> tuple[object, ...]:
    """``ldmatrix ... .trans``: lane (g, c) receives rows 2c, 2c+1 of column g."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),  # pyrefly: ignore[missing-attribute]
        [cutlass.Int32(smem_address).ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return _words(result, 4, loc=loc, ip=ip)


@dsl_user_op
def movmatrix_trans(
    word: object, *, loc: object = None, ip: object = None
) -> cutlass.Uint32:
    """``movmatrix.sync.aligned.m8n8.trans.b16``: transpose one 8x8 16-bit
    tile held in the mma C/A fragment layout (lane (g, c) holds row g,
    columns 2c, 2c+1), yielding the mma B fragment of the transposed tile."""
    result = llvm.inline_asm(
        T.i32(),
        [cutlass.Uint32(word).ir_value(loc=loc, ip=ip)],
        "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;",
        "=r,r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return cutlass.Uint32(result)


def _mma_m16n8k16(
    kind: str,
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
    loc: object,
    ip: object,
) -> tuple[object, ...]:
    values = [
        *[cutlass.Uint32(v).ir_value(loc=loc, ip=ip) for v in (a0, a1, a2, a3, b0, b1)],
        *[cutlass.Float32(v).ir_value(loc=loc, ip=ip) for v in (c0, c1, c2, c3)],
    ]
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4),  # pyrefly: ignore[missing-attribute]
        values,
        f"mma.sync.aligned.m16n8k16.row.col.f32.{kind}.{kind}.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return _floats(result, 4, loc=loc, ip=ip)


@dsl_user_op
def mma_bf16(
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
    """``mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32``."""
    return _mma_m16n8k16("bf16", a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, loc=loc, ip=ip)


@dsl_user_op
def mma_f16(
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
    """``mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32``."""
    return _mma_m16n8k16("f16", a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, loc=loc, ip=ip)


def _pack_x2(
    kind: str, low: object, high: object, *, loc: object, ip: object
) -> cutlass.Uint32:
    result = llvm.inline_asm(
        T.i32(),
        [
            cutlass.Float32(high).ir_value(loc=loc, ip=ip),
            cutlass.Float32(low).ir_value(loc=loc, ip=ip),
        ],
        f"cvt.rn.{kind}x2.f32 $0, $1, $2;",
        "=r,f,f",
        has_side_effects=False,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return cutlass.Uint32(result)


@dsl_user_op
def pack_bf16x2(
    low: object, high: object, *, loc: object = None, ip: object = None
) -> cutlass.Uint32:
    """``cvt.rn.bf16x2.f32``: ``low`` in the low half, ``high`` in the high half."""
    return _pack_x2("bf16", low, high, loc=loc, ip=ip)


@dsl_user_op
def pack_f16x2(
    low: object, high: object, *, loc: object = None, ip: object = None
) -> cutlass.Uint32:
    """``cvt.rn.f16x2.f32``: ``low`` in the low half, ``high`` in the high half."""
    return _pack_x2("f16", low, high, loc=loc, ip=ip)


@dsl_user_op
def sts_f32(
    smem_address: object, value: object, *, loc: object = None, ip: object = None
) -> None:
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(smem_address).ir_value(loc=loc, ip=ip),
            cutlass.Float32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.f32 [$0], $1;",
        "r,f,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )


@dsl_user_op
def lds_f32(
    smem_address: object, *, loc: object = None, ip: object = None
) -> cutlass.Float32:
    result = llvm.inline_asm(
        T.f32(),
        [cutlass.Int32(smem_address).ir_value(loc=loc, ip=ip)],
        "ld.shared.f32 $0, [$1];",
        "=f,r,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return cutlass.Float32(result)


@dsl_user_op
def lds_v4_f32(
    smem_address: object, *, loc: object = None, ip: object = None
) -> tuple[object, ...]:
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4),  # pyrefly: ignore[missing-attribute]
        [cutlass.Int32(smem_address).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return _floats(result, 4, loc=loc, ip=ip)


@dsl_user_op
def stg_v4_b32(
    gmem_address: object,
    w0: object,
    w1: object,
    w2: object,
    w3: object,
    *,
    loc: object = None,
    ip: object = None,
) -> None:
    llvm.inline_asm(
        None,
        [
            cutlass.Int64(gmem_address).ir_value(loc=loc, ip=ip),
            *[cutlass.Uint32(v).ir_value(loc=loc, ip=ip) for v in (w0, w1, w2, w3)],
        ],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )


@dsl_user_op
def stg_f32(
    gmem_address: object, value: object, *, loc: object = None, ip: object = None
) -> None:
    llvm.inline_asm(
        None,
        [
            cutlass.Int64(gmem_address).ir_value(loc=loc, ip=ip),
            cutlass.Float32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.f32 [$0], $1;",
        "l,f,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )


def octet_max(value: object) -> object:
    """Max over the eight lanes that share ``lane % 4`` (xor 4, 8, 16): the
    lanes holding one query's scores in the transposed S^T fragment."""
    for offset in (4, 8, 16):
        value = cute.arch.fmax(value, cute.arch.shuffle_sync_bfly(value, offset))
    return value


def octet_sum(value: object) -> object:
    """Sum over the eight lanes that share ``lane % 4``, in a fixed order."""
    for offset in (4, 8, 16):
        value = value + cute.arch.shuffle_sync_bfly(value, offset)
    return value


def lane_group_sum(value: object, width: int) -> object:
    """Sum over the ``width`` consecutive lanes (a power of two dividing 32)
    that share one output row in the combine epilogue.

    A butterfly over the offsets 1, 2, ..., width/2 in a fixed order: fp32
    addition is commutative, so every lane of the group ends with the same
    value and repeated launches are bit-identical."""
    offset = 1
    while offset < width:
        value = value + cute.arch.shuffle_sync_bfly(value, offset)
        offset *= 2
    return value


def lane_group_max(value: object, width: int) -> object:
    """NaN-propagating max over the ``width`` consecutive lanes of a row."""
    offset = 1
    while offset < width:
        value = cute.math.max(
            value, cute.arch.shuffle_sync_bfly(value, offset), propagate_nan=True
        )
        offset *= 2
    return value


def lane_group_min(value: object, width: int) -> object:
    """NaN-propagating min over the ``width`` consecutive lanes of a row."""
    offset = 1
    while offset < width:
        value = cute.math.min(
            value, cute.arch.shuffle_sync_bfly(value, offset), propagate_nan=True
        )
        offset *= 2
    return value
