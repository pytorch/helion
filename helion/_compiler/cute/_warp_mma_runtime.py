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

# ``ld.global`` qualifier of a load: the read-only (non-coherent) path for a
# tensor no store of the launch writes, else the default path.
_NC = {False: "", True: ".nc"}


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
def mma_tf32(
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
    """``mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32``: lane (g, c)
    supplies A rows g / g+8 at k c (a0 / a1) and k c+4 (a2 / a3), column g
    of B at the same k (b0 / b1), as fp32 bit patterns the tensor core
    truncates to tf32 (the ``input_precision='tf32'`` numerics)."""
    values = [
        *[cutlass.Uint32(v).ir_value(loc=loc, ip=ip) for v in (a0, a1, a2, a3, b0, b1)],
        *[cutlass.Float32(v).ir_value(loc=loc, ip=ip) for v in (c0, c1, c2, c3)],
    ]
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4),  # pyrefly: ignore[missing-attribute]
        values,
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return _floats(result, 4, loc=loc, ip=ip)


def _cvt_i16x8_sts(
    source: str,
    kind: str,
    smem_address: object,
    w0: object,
    w1: object,
    w2: object,
    w3: object,
    *,
    loc: object,
    ip: object,
) -> None:
    # Eight 16-bit integers (``source``: s16 / u16) widened exactly to fp32
    # and rounded (RN, the program's own ``.to()`` rounding) to eight half
    # words (``kind``: bf16 / f16), stored as one 16-byte shared packet.
    unpack = " ".join(
        f"mov.b32 {{i{2 * j}, i{2 * j + 1}}}, ${j + 1};" for j in range(4)
    )
    widen = " ".join(f"cvt.rn.f32.{source} f{j}, i{j};" for j in range(8))
    narrow = " ".join(f"cvt.rn.{kind}.f32 h{j}, f{j};" for j in range(8))
    pack = " ".join(f"mov.b32 o{j}, {{h{2 * j}, h{2 * j + 1}}};" for j in range(4))
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(smem_address).ir_value(loc=loc, ip=ip),
            *[cutlass.Uint32(w).ir_value(loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
        ],
        "{ .reg .b16 i<8>; .reg .f32 f<8>; .reg .b16 h<8>; .reg .b32 o<4>; "
        f"{unpack} {widen} {narrow} {pack} "
        "st.shared.v4.b32 [$0], {o0, o1, o2, o3}; }",
        "r,r,r,r,r,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )


@dsl_user_op
def cvt_s16x8_bf16_sts(
    smem_address: object,
    w0: object,
    w1: object,
    w2: object,
    w3: object,
    *,
    loc: object = None,
    ip: object = None,
) -> None:
    """Eight int16 words -> eight bf16 (RN), stored as one shared packet."""
    _cvt_i16x8_sts("s16", "bf16", smem_address, w0, w1, w2, w3, loc=loc, ip=ip)


@dsl_user_op
def cvt_s16x8_f16_sts(
    smem_address: object,
    w0: object,
    w1: object,
    w2: object,
    w3: object,
    *,
    loc: object = None,
    ip: object = None,
) -> None:
    """Eight int16 words -> eight fp16 (RN), stored as one shared packet."""
    _cvt_i16x8_sts("s16", "f16", smem_address, w0, w1, w2, w3, loc=loc, ip=ip)


@dsl_user_op
def cvt_u16x8_bf16_sts(
    smem_address: object,
    w0: object,
    w1: object,
    w2: object,
    w3: object,
    *,
    loc: object = None,
    ip: object = None,
) -> None:
    """Eight uint16 words -> eight bf16 (RN), stored as one shared packet."""
    _cvt_i16x8_sts("u16", "bf16", smem_address, w0, w1, w2, w3, loc=loc, ip=ip)


@dsl_user_op
def cvt_u16x8_f16_sts(
    smem_address: object,
    w0: object,
    w1: object,
    w2: object,
    w3: object,
    *,
    loc: object = None,
    ip: object = None,
) -> None:
    """Eight uint16 words -> eight fp16 (RN), stored as one shared packet."""
    _cvt_i16x8_sts("u16", "f16", smem_address, w0, w1, w2, w3, loc=loc, ip=ip)


@dsl_user_op
def ldg_b32(
    gmem_address: object, *, nc: bool = False, loc: object = None, ip: object = None
) -> cutlass.Uint32:
    """``ld.global.b32``: one 4-byte word (``nc``: the read-only path)."""
    result = llvm.inline_asm(
        T.i32(),
        [cutlass.Int64(gmem_address).ir_value(loc=loc, ip=ip)],
        f"ld.global{_NC[nc]}.b32 $0, [$1];",
        "=r,l,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return cutlass.Uint32(result)


@dsl_user_op
def cp_async_16_ca(
    smem_address: object, gmem_address: object, *, loc: object = None, ip: object = None
) -> None:
    """``cp.async.ca.shared.global`` of one 16-byte packet (L1-allocating)."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(smem_address).ir_value(loc=loc, ip=ip),
            cutlass.Int64(gmem_address).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.ca.shared.global [$0], [$1], 16;",
        "r,l,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )


@dsl_user_op
def cp_async_4(
    smem_address: object, gmem_address: object, *, loc: object = None, ip: object = None
) -> None:
    """``cp.async.ca.shared.global`` of one 4-byte word (a side-load element
    or a pair of 16-bit ones) into a per-thread shared slot."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(smem_address).ir_value(loc=loc, ip=ip),
            cutlass.Int64(gmem_address).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.ca.shared.global [$0], [$1], 4;",
        "r,l,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )


@dsl_user_op
def lds_s32(
    smem_address: object, *, loc: object = None, ip: object = None
) -> cutlass.Int32:
    """``ld.shared.s32``: one int32 value."""
    result = llvm.inline_asm(
        T.i32(),
        [cutlass.Int32(smem_address).ir_value(loc=loc, ip=ip)],
        "ld.shared.s32 $0, [$1];",
        "=r,r,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return cutlass.Int32(result)


def _lds_b16_as_f32(
    kind: str, smem_address: object, *, loc: object, ip: object
) -> cutlass.Float32:
    result = llvm.inline_asm(
        T.f32(),
        [cutlass.Int32(smem_address).ir_value(loc=loc, ip=ip)],
        f"{{ .reg .b16 h; ld.shared.b16 h, [$1]; cvt.f32.{kind} $0, h; }}",
        "=f,r,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return cutlass.Float32(result)


@dsl_user_op
def lds_f16_as_f32(
    smem_address: object, *, loc: object = None, ip: object = None
) -> cutlass.Float32:
    """``ld.shared.b16`` of an fp16 element, widened to fp32 (exact)."""
    return _lds_b16_as_f32("f16", smem_address, loc=loc, ip=ip)


@dsl_user_op
def lds_bf16_as_f32(
    smem_address: object, *, loc: object = None, ip: object = None
) -> cutlass.Float32:
    """``ld.shared.b16`` of a bf16 element, widened to fp32 (exact)."""
    return _lds_b16_as_f32("bf16", smem_address, loc=loc, ip=ip)


@dsl_user_op
def ldg_v4_b32(
    gmem_address: object, *, loc: object = None, ip: object = None
) -> tuple[object, ...]:
    """``ld.global.v4.b32``: one 16-byte packet (eight 16-bit elements) as
    four words."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),  # pyrefly: ignore[missing-attribute]
        [cutlass.Int64(gmem_address).ir_value(loc=loc, ip=ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return _words(result, 4, loc=loc, ip=ip)


@dsl_user_op
def ldg_f32(
    gmem_address: object, *, nc: bool = False, loc: object = None, ip: object = None
) -> cutlass.Float32:
    """``ld.global.f32``: one fp32 value (``nc``: the read-only path)."""
    result = llvm.inline_asm(
        T.f32(),
        [cutlass.Int64(gmem_address).ir_value(loc=loc, ip=ip)],
        f"ld.global{_NC[nc]}.f32 $0, [$1];",
        "=f,l,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return cutlass.Float32(result)


@dsl_user_op
def ldg_v2_f32(
    gmem_address: object, *, nc: bool = False, loc: object = None, ip: object = None
) -> tuple[object, ...]:
    """``ld.global.v2.f32``: two consecutive fp32 values (8-byte aligned)."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 2),  # pyrefly: ignore[missing-attribute]
        [cutlass.Int64(gmem_address).ir_value(loc=loc, ip=ip)],
        f"ld.global{_NC[nc]}.v2.f32 {{$0, $1}}, [$2];",
        "=f,=f,l,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return _floats(result, 2, loc=loc, ip=ip)


def _ldg_b16_as_f32(
    kind: str, gmem_address: object, *, nc: bool, loc: object, ip: object
) -> cutlass.Float32:
    result = llvm.inline_asm(
        T.f32(),
        [cutlass.Int64(gmem_address).ir_value(loc=loc, ip=ip)],
        f"{{ .reg .b16 h; ld.global{_NC[nc]}.b16 h, [$1]; cvt.f32.{kind} $0, h; }}",
        "=f,l,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return cutlass.Float32(result)


@dsl_user_op
def ldg_f16_as_f32(
    gmem_address: object, *, nc: bool = False, loc: object = None, ip: object = None
) -> cutlass.Float32:
    """One fp16 element loaded and widened to fp32 (exact)."""
    return _ldg_b16_as_f32("f16", gmem_address, nc=nc, loc=loc, ip=ip)


@dsl_user_op
def ldg_bf16_as_f32(
    gmem_address: object, *, nc: bool = False, loc: object = None, ip: object = None
) -> cutlass.Float32:
    """One bf16 element loaded and widened to fp32 (exact)."""
    return _ldg_b16_as_f32("bf16", gmem_address, nc=nc, loc=loc, ip=ip)


def _unpack_x2(
    kind: str, word: object, *, loc: object, ip: object
) -> tuple[object, ...]:
    if kind == "bf16":
        # A bf16 is the high half of its fp32: the low element shifts up,
        # the high element masks (two integer ops; ``cvt.f32.bf16`` of the
        # halves lowers to a PRMT plus a shift for the high half).  Exact.
        asm = (
            "{ .reg .b32 lo, hi; shl.b32 lo, $2, 16; and.b32 hi, $2, 0xffff0000; "
            "mov.b32 $0, lo; mov.b32 $1, hi; }"
        )
    else:
        asm = (
            "{ .reg .b16 lo, hi; mov.b32 {lo, hi}, $2; "
            f"cvt.f32.{kind} $0, lo; cvt.f32.{kind} $1, hi; }}"
        )
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 2),  # pyrefly: ignore[missing-attribute]
        [cutlass.Uint32(word).ir_value(loc=loc, ip=ip)],
        asm,
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
def round_bf16_f32(
    value: object, *, loc: object = None, ip: object = None
) -> cutlass.Float32:
    """``value`` rounded to bf16 (round to nearest even) and widened back to
    fp32: the bits of ``float32(bfloat16(value))``.  ``cvt.rn.bf16x2.f32``
    of the value with itself puts the rounded half-word in the high half
    (``F2FP.BF16.F32.PACK_AB``, an FMA-pipe instruction) and the mask is the
    exact widening; the scalar ``cvt.rn.bf16.f32`` the DSL's cast emits
    runs on the conversion unit (``F2F``, the MIO queue) and its widening
    is a shift behind it."""
    result = llvm.inline_asm(
        T.f32(),
        [cutlass.Float32(value).ir_value(loc=loc, ip=ip)],
        "{ .reg .b32 w; cvt.rn.bf16x2.f32 w, $1, $1; "
        "and.b32 w, w, 0xffff0000; mov.b32 $0, w; }",
        "=f,f",
        has_side_effects=False,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return cutlass.Float32(result)


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


# ---------------------------------------------------------------------------
# Register-resident sub-block programs (``cute_register_subblocks``, see
# ``cute_register_subblocks.py``).  Two flavours of global load: the
# ``_ro`` helpers have no side effects and no memory clobber (a tensor no
# store of the launch writes: the compiler may schedule them freely, which
# is what the explicit prefetch order relies on), the ordered ones keep
# program order (outputs read back before a read-modify-write).  The
# predicated loads zero-fill under a false predicate (the varlen window
# rows past a sequence's end).


def _ldg_vector(
    gmem_address: object,
    count: int,
    kind: str,
    *,
    ordered: bool,
    loc: object,
    ip: object,
) -> tuple[object, ...]:
    regs = ", ".join(f"${i}" for i in range(count))
    constraint = ("=f," if kind == "f32" else "=r,") * count + "l"
    vec = f".v{count}" if count > 1 else ""
    braces = f"{{{regs}}}" if count > 1 else regs
    result = llvm.inline_asm(
        llvm.StructType.get_literal(  # pyrefly: ignore[missing-attribute]
            [T.f32() if kind == "f32" else T.i32()] * count
        )
        if count > 1
        else (T.f32() if kind == "f32" else T.i32()),
        [cutlass.Int64(gmem_address).ir_value(loc=loc, ip=ip)],
        f"ld.global{vec}.{kind} {braces}, [${count}];",
        constraint + (",~{memory}" if ordered else ""),
        has_side_effects=ordered,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    if count == 1:
        return (cutlass.Float32(result) if kind == "f32" else cutlass.Uint32(result),)
    return (
        _floats(result, count, loc=loc, ip=ip)
        if kind == "f32"
        else _words(result, count, loc=loc, ip=ip)
    )


@dsl_user_op
def ldg_v4_f32_ro(
    gmem_address: object, *, loc: object = None, ip: object = None
) -> tuple[object, ...]:
    """``ld.global.v4.f32`` of a tensor the launch never writes (pure)."""
    return _ldg_vector(gmem_address, 4, "f32", ordered=False, loc=loc, ip=ip)


@dsl_user_op
def ldg_v2_f32_ro(
    gmem_address: object, *, loc: object = None, ip: object = None
) -> tuple[object, ...]:
    """``ld.global.v2.f32`` of a tensor the launch never writes (pure)."""
    return _ldg_vector(gmem_address, 2, "f32", ordered=False, loc=loc, ip=ip)


@dsl_user_op
def ldg_f32_ro(
    gmem_address: object, *, loc: object = None, ip: object = None
) -> cutlass.Float32:
    """``ld.global.f32`` of a tensor the launch never writes (pure)."""
    return _ldg_vector(gmem_address, 1, "f32", ordered=False, loc=loc, ip=ip)[0]  # pyrefly: ignore[bad-return]


@dsl_user_op
def ldg_v4_b32_ro(
    gmem_address: object, *, loc: object = None, ip: object = None
) -> tuple[object, ...]:
    """``ld.global.v4.b32`` of a tensor the launch never writes (pure)."""
    return _ldg_vector(gmem_address, 4, "b32", ordered=False, loc=loc, ip=ip)


@dsl_user_op
def ldg_v2_b32_ro(
    gmem_address: object, *, loc: object = None, ip: object = None
) -> tuple[object, ...]:
    """``ld.global.v2.b32`` of a tensor the launch never writes (pure)."""
    return _ldg_vector(gmem_address, 2, "b32", ordered=False, loc=loc, ip=ip)


@dsl_user_op
def ldg_b32_ro(
    gmem_address: object, *, loc: object = None, ip: object = None
) -> cutlass.Uint32:
    """``ld.global.b32`` of a tensor the launch never writes (pure)."""
    return _ldg_vector(gmem_address, 1, "b32", ordered=False, loc=loc, ip=ip)[0]  # pyrefly: ignore[bad-return]


@dsl_user_op
def ldg_v4_f32(
    gmem_address: object, *, loc: object = None, ip: object = None
) -> tuple[object, ...]:
    """``ld.global.v4.f32`` kept in program order (an output read back)."""
    return _ldg_vector(gmem_address, 4, "f32", ordered=True, loc=loc, ip=ip)


@dsl_user_op
def ldg_v4_b32_ord(
    gmem_address: object, *, loc: object = None, ip: object = None
) -> tuple[object, ...]:
    """``ld.global.v4.b32`` kept in program order."""
    return _ldg_vector(gmem_address, 4, "b32", ordered=True, loc=loc, ip=ip)


@dsl_user_op
def ldg_v2_b32(
    gmem_address: object, *, loc: object = None, ip: object = None
) -> tuple[object, ...]:
    """``ld.global.v2.b32`` kept in program order."""
    return _ldg_vector(gmem_address, 2, "b32", ordered=True, loc=loc, ip=ip)


def _ldg_vector_pred(
    gmem_address: object,
    pred: object,
    count: int,
    kind: str,
    *,
    loc: object,
    ip: object,
) -> tuple[object, ...]:
    regs = ", ".join(f"${i}" for i in range(count))
    zero = "0f00000000" if kind == "f32" else "0"
    zeros = " ".join(f"mov.{kind} ${i}, {zero};" for i in range(count))
    vec = f".v{count}" if count > 1 else ""
    braces = f"{{{regs}}}" if count > 1 else regs
    constraint = ("=f," if kind == "f32" else "=r,") * count + "l,r,~{memory}"
    result = llvm.inline_asm(
        llvm.StructType.get_literal(  # pyrefly: ignore[missing-attribute]
            [T.f32() if kind == "f32" else T.i32()] * count
        )
        if count > 1
        else (T.f32() if kind == "f32" else T.i32()),
        [
            cutlass.Int64(gmem_address).ir_value(loc=loc, ip=ip),
            cutlass.Int32(pred).ir_value(loc=loc, ip=ip),
        ],
        f"{{ .reg .pred p; setp.ne.s32 p, ${count + 1}, 0; {zeros} "
        f"@p ld.global{vec}.{kind} {braces}, [${count}]; }}",
        constraint,
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    if count == 1:
        return (cutlass.Float32(result) if kind == "f32" else cutlass.Uint32(result),)
    return (
        _floats(result, count, loc=loc, ip=ip)
        if kind == "f32"
        else _words(result, count, loc=loc, ip=ip)
    )


@dsl_user_op
def ldg_v4_f32_pred(
    gmem_address: object, pred: object, *, loc: object = None, ip: object = None
) -> tuple[object, ...]:
    """``ld.global.v4.f32`` under ``pred`` (four zeros when it is 0)."""
    return _ldg_vector_pred(gmem_address, pred, 4, "f32", loc=loc, ip=ip)


@dsl_user_op
def ldg_v4_b32_pred(
    gmem_address: object, pred: object, *, loc: object = None, ip: object = None
) -> tuple[object, ...]:
    """``ld.global.v4.b32`` under ``pred`` (four zero words when it is 0)."""
    return _ldg_vector_pred(gmem_address, pred, 4, "b32", loc=loc, ip=ip)


@dsl_user_op
def ldg_v2_b32_pred(
    gmem_address: object, pred: object, *, loc: object = None, ip: object = None
) -> tuple[object, ...]:
    """``ld.global.v2.b32`` under ``pred`` (two zero words when it is 0)."""
    return _ldg_vector_pred(gmem_address, pred, 2, "b32", loc=loc, ip=ip)


@dsl_user_op
def stg_v4_f32(
    gmem_address: object,
    x0: object,
    x1: object,
    x2: object,
    x3: object,
    *,
    loc: object = None,
    ip: object = None,
) -> None:
    """``st.global.v4.f32``: four consecutive fp32 values (16-byte aligned)."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int64(gmem_address).ir_value(loc=loc, ip=ip),
            *[cutlass.Float32(x).ir_value(loc=loc, ip=ip) for x in (x0, x1, x2, x3)],
        ],
        "st.global.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,f,f,f,f,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )


@dsl_user_op
def stg_v2_b32(
    gmem_address: object,
    w0: object,
    w1: object,
    *,
    loc: object = None,
    ip: object = None,
) -> None:
    """``st.global.v2.b32``: two 4-byte words (8-byte aligned; four 16-bit values)."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int64(gmem_address).ir_value(loc=loc, ip=ip),
            *[cutlass.Uint32(w).ir_value(loc=loc, ip=ip) for w in (w0, w1)],
        ],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )


def _sel_cmp(
    op: str, a: object, b: object, x: object, y: object, *, loc: object, ip: object
) -> cutlass.Float32:
    result = llvm.inline_asm(
        T.f32(),
        [
            cutlass.Int32(a).ir_value(loc=loc, ip=ip),
            cutlass.Int32(b).ir_value(loc=loc, ip=ip),
            cutlass.Float32(x).ir_value(loc=loc, ip=ip),
            cutlass.Float32(y).ir_value(loc=loc, ip=ip),
        ],
        f"{{ .reg .pred p; setp.{op}.s32 p, $1, $2; selp.f32 $0, $3, $4, p; }}",
        "=f,r,r,f,f",
        has_side_effects=False,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return cutlass.Float32(result)


@dsl_user_op
def sel_ge_f32(
    a: object, b: object, x: object, y: object, *, loc: object = None, ip: object = None
) -> cutlass.Float32:
    """``x if a >= b else y`` (``setp`` + ``selp``): the inclusive triangular mask."""
    return _sel_cmp("ge", a, b, x, y, loc=loc, ip=ip)


@dsl_user_op
def sel_gt_f32(
    a: object, b: object, x: object, y: object, *, loc: object = None, ip: object = None
) -> cutlass.Float32:
    """``x if a > b else y``: the strict triangular mask."""
    return _sel_cmp("gt", a, b, x, y, loc=loc, ip=ip)


@dsl_user_op
def sel_le_f32(
    a: object, b: object, x: object, y: object, *, loc: object = None, ip: object = None
) -> cutlass.Float32:
    """``x if a <= b else y``."""
    return _sel_cmp("le", a, b, x, y, loc=loc, ip=ip)


@dsl_user_op
def sel_lt_f32(
    a: object, b: object, x: object, y: object, *, loc: object = None, ip: object = None
) -> cutlass.Float32:
    """``x if a < b else y``."""
    return _sel_cmp("lt", a, b, x, y, loc=loc, ip=ip)


@dsl_user_op
def sel_eq_f32(
    a: object, b: object, x: object, y: object, *, loc: object = None, ip: object = None
) -> cutlass.Float32:
    """``x if a == b else y``."""
    return _sel_cmp("eq", a, b, x, y, loc=loc, ip=ip)


@dsl_user_op
def sel_ne_f32(
    a: object, b: object, x: object, y: object, *, loc: object = None, ip: object = None
) -> cutlass.Float32:
    """``x if a != b else y``."""
    return _sel_cmp("ne", a, b, x, y, loc=loc, ip=ip)
