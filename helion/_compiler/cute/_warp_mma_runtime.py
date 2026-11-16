# Runtime support for the register-MMA GEMM family (``cute_matmul_family=
# "warp_mma"``, see ``cute_warp_mma_gemm.py``).
#
# NOTE: deliberately NO ``from __future__ import annotations`` (see
# _flash_runtime.py): these helpers are imported by the generated module and
# traced inline into the kernel body.  Every helper is one PTX instruction, so
# the emitted body is a straight line of SSA values.  The single-instruction
# helpers are ``device/ptx.py`` rows and the MMA / pack helpers shared with
# the flash row programs come from their runtime module, re-exported.  The
# global loads carry a memory clobber (like the stores) so an aux row that
# aliases the output (an in-place residual) is read before it is written.
import cutlass
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T
from cutlass.cutlass_dsl import dsl_user_op

from ._flash_row_mma_runtime import mma_bf16  # noqa: F401
from ._flash_row_mma_runtime import mma_f16  # noqa: F401
from ._flash_row_mma_runtime import pack_bf16x2  # noqa: F401
from ._flash_row_mma_runtime import pack_f16x2  # noqa: F401
from .device.ptx import ASM_KW
from .device.ptx import cp_async_4  # noqa: F401
from .device.ptx import cp_async_16  # noqa: F401
from .device.ptx import cp_async_16_ca  # noqa: F401
from .device.ptx import floats
from .device.ptx import ldmatrix_x2  # noqa: F401
from .device.ptx import ldmatrix_x2_trans  # noqa: F401
from .device.ptx import ldmatrix_x4  # noqa: F401
from .device.ptx import ldmatrix_x4_trans  # noqa: F401
from .device.ptx import lds_bf16_as_f32  # noqa: F401
from .device.ptx import lds_f16_as_f32  # noqa: F401
from .device.ptx import lds_s32  # noqa: F401
from .device.ptx import lds_v4_b32  # noqa: F401
from .device.ptx import mma_e4m3  # noqa: F401
from .device.ptx import mma_tf32  # noqa: F401
from .device.ptx import round_bf16_f32  # noqa: F401
from .device.ptx import sel_eq_f32  # noqa: F401
from .device.ptx import sel_ge_f32  # noqa: F401
from .device.ptx import sel_gt_f32  # noqa: F401
from .device.ptx import sel_le_f32  # noqa: F401
from .device.ptx import sel_lt_f32  # noqa: F401
from .device.ptx import sel_ne_f32  # noqa: F401
from .device.ptx import stg_b32  # noqa: F401
from .device.ptx import stg_bf16  # noqa: F401
from .device.ptx import stg_f16  # noqa: F401
from .device.ptx import stg_f32  # noqa: F401
from .device.ptx import stg_v2_b32  # noqa: F401
from .device.ptx import stg_v2_f32  # noqa: F401
from .device.ptx import stg_v4_b32  # noqa: F401
from .device.ptx import stg_v4_f32  # noqa: F401
from .device.ptx import sts_v4_b32  # noqa: F401
from .device.ptx import unpack_f16x2  # noqa: F401
from .device.ptx import words

# ``ld.global`` qualifier of a load: the read-only (non-coherent) path for a
# tensor no store of the launch writes, else the default path.
_NC = {False: "", True: ".nc"}


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
        **ASM_KW,
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
        **ASM_KW,
    )
    return cutlass.Uint32(result)


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
        **ASM_KW,
    )
    return words(result, 4, loc=loc, ip=ip)


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
        **ASM_KW,
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
        **ASM_KW,
    )
    return floats(result, 2, loc=loc, ip=ip)


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
        **ASM_KW,
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


@dsl_user_op
def unpack_bf16x2(
    word: object, *, loc: object = None, ip: object = None
) -> tuple[object, ...]:
    """``(low, high)`` fp32 values of a packed bf16x2 word (exact)."""
    # A bf16 is the high half of its fp32: the low element shifts up, the
    # high element masks (two integer ops; ``cvt.f32.bf16`` of the halves
    # lowers to a PRMT plus a shift for the high half).
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 2),  # pyrefly: ignore[missing-attribute]
        [cutlass.Uint32(word).ir_value(loc=loc, ip=ip)],
        "{ .reg .b32 lo, hi; shl.b32 lo, $2, 16; and.b32 hi, $2, 0xffff0000; "
        "mov.b32 $0, lo; mov.b32 $1, hi; }",
        "=f,=f,r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
        **ASM_KW,
    )
    return floats(result, 2, loc=loc, ip=ip)


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
        **ASM_KW,
    )
    if count == 1:
        return (cutlass.Float32(result) if kind == "f32" else cutlass.Uint32(result),)
    return (
        floats(result, count, loc=loc, ip=ip)
        if kind == "f32"
        else words(result, count, loc=loc, ip=ip)
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
        **ASM_KW,
    )
    if count == 1:
        return (cutlass.Float32(result) if kind == "f32" else cutlass.Uint32(result),)
    return (
        floats(result, count, loc=loc, ip=ip)
        if kind == "f32"
        else words(result, count, loc=loc, ip=ip)
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
